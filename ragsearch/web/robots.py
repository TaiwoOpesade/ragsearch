"""A small robots.txt matcher that follows RFC 9309 (the Robots Exclusion Protocol).

Python's ``urllib.robotparser`` uses the first matching rule and ignores the
``*`` and ``$`` wildcards, so it can allow pages a site disallows. This module
implements what the RFC requires:

* the group(s) naming our product token apply, else the ``*`` group(s), merged;
* the longest (most specific) matching rule wins, and ``Allow`` wins a tie;
* ``*`` matches any sequence and a trailing ``$`` anchors the end of the path;
* ``/robots.txt`` itself is always allowed.

Rules and paths are compared in one canonical percent-encoded form, so ``%2A`` and
``%24`` stay literal characters and ``%0A`` cannot slip past a wildcard. Matching
uses no regular expressions: a greedy scan runs in linear time, so a hostile
robots.txt cannot stall the crawler.
"""

import string
from dataclasses import dataclass

_UNRESERVED = frozenset(string.ascii_letters + string.digits + "-._~")
# Characters a URL may carry unencoded: unreserved plus RFC 3986's reserved set.
_SAFE = _UNRESERVED | frozenset(":/?#[]@!&'()+,;=")
_HEX = frozenset(string.hexdigits)


def _canonical(text: str, literal_specials: bool) -> str:
    """Percent-encode ``text`` the one canonical way (RFC 3986 normalisation).

    This matches what ``requests`` actually sends: escapes of unreserved characters
    are decoded, other escapes get upper-case hex, and everything else outside the
    safe set (spaces, ``|``, a stray ``%``, non-ASCII...) is encoded as UTF-8 bytes.
    In a rule (``literal_specials`` False) ``*`` and a final ``$`` stay special; in a
    URL path they are data, encoded the way a rule must write them.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        ch, escape = text[i], text[i + 1 : i + 3]
        if ch == "%" and len(escape) == 2 and set(escape) <= _HEX:
            decoded = chr(int(escape, 16))
            out.append(decoded if decoded in _UNRESERVED else "%" + escape.upper())
            i += 3
            continue
        special = not literal_specials and (ch == "*" or (ch == "$" and i == len(text) - 1))
        if special or ch in _SAFE:
            out.append(ch)
        else:
            out.append("".join(f"%{b:02X}" for b in ch.encode("utf-8", "surrogatepass")))
        i += 1
    return "".join(out)


@dataclass(frozen=True)
class _Rule:
    allow: bool
    length: int
    parts: tuple[str, ...]  # the pattern split on "*"
    anchored: bool  # the pattern ended with "$"

    @classmethod
    def build(cls, allow: bool, path: str) -> "_Rule":
        pattern = _canonical(path, literal_specials=False)
        anchored = pattern.endswith("$")
        body = pattern[:-1] if anchored else pattern
        return cls(allow, len(pattern), tuple(body.split("*")), anchored)

    def matches(self, path: str) -> bool:
        first, *rest = self.parts
        if not path.startswith(first):
            return False
        if not rest:
            return path == first if self.anchored else True
        pos = len(first)
        for middle in rest[:-1]:  # leftmost match of each piece leaves the most room
            found = path.find(middle, pos)
            if found < 0:
                return False
            pos = found + len(middle)
        last = rest[-1]
        if self.anchored:
            return len(path) - len(last) >= pos and path.endswith(last)
        return path.find(last, pos) >= 0


def _parse_groups(text: str) -> list[tuple[list[str], list[tuple[bool, str]]]]:
    groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] | None = None
    for raw in text.removeprefix("﻿").splitlines():
        line = raw.split("#", 1)[0].strip()
        key, sep, value = line.partition(":")
        if not sep:
            continue
        key, value = key.strip().lower(), value.strip()
        if key == "user-agent":
            if rules is not None:  # a user-agent line after rules starts a new group
                groups.append((agents, rules))
                agents, rules = [], None
            agents.append(value.lower())
        elif key in ("allow", "disallow") and agents:
            rules = [] if rules is None else rules
            if value:  # an empty Disallow (or Allow) means "no restriction"
                rules.append((key == "allow", value))
    if agents:
        groups.append((agents, rules or []))
    return groups


class RobotsRules:
    def __init__(self, rules: list[_Rule], default: bool = True):
        self._rules = rules
        self._default = default

    @classmethod
    def allow_all(cls) -> "RobotsRules":
        return cls([], default=True)

    @classmethod
    def disallow_all(cls) -> "RobotsRules":
        return cls([], default=False)

    @classmethod
    def parse(cls, text: str, user_agent: str) -> "RobotsRules":
        token = user_agent.split("/", 1)[0].strip().lower()
        groups = _parse_groups(text)
        ours = [r for agents, r in groups if any(a.split("/", 1)[0] == token for a in agents)]
        chosen = ours or [r for agents, r in groups if "*" in agents]
        return cls([_Rule.build(allow, path) for group in chosen for allow, path in group])

    def allowed(self, path: str) -> bool:
        """Whether ``path`` (URL path plus any ``?query``) may be fetched."""
        path = _canonical(path or "/", literal_specials=True)
        if path == "/robots.txt":
            return True
        best: _Rule | None = None
        for rule in self._rules:
            if rule.matches(path) and (
                best is None
                or rule.length > best.length
                or (rule.length == best.length and rule.allow)
            ):
                best = rule
        return self._default if best is None else best.allow
