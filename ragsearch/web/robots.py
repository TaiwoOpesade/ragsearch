"""A small robots.txt matcher that follows RFC 9309 (the Robots Exclusion Protocol).

Python's ``urllib.robotparser`` uses the first matching rule and ignores the
``*`` and ``$`` wildcards, so it can allow pages a site disallows. This module
implements what the RFC requires:

* the group(s) naming our product token apply, else the ``*`` group(s), merged;
* the longest (most specific) matching rule wins, and ``Allow`` wins a tie;
* ``*`` matches any sequence and a trailing ``$`` anchors the end of the path;
* ``/robots.txt`` itself is always allowed.
"""

import re
from dataclasses import dataclass
from urllib.parse import unquote


@dataclass(frozen=True)
class _Rule:
    allow: bool
    length: int
    pattern: re.Pattern


def _compile(path: str) -> re.Pattern:
    path = unquote(path)
    anchored = path.endswith("$")
    body = path[:-1] if anchored else path
    regex = ".*".join(re.escape(part) for part in body.split("*"))
    return re.compile(regex + (r"\Z" if anchored else ""))


def _parse_groups(text: str) -> list[tuple[list[str], list[tuple[bool, str]]]]:
    groups: list[tuple[list[str], list[tuple[bool, str]]]] = []
    agents: list[str] = []
    rules: list[tuple[bool, str]] | None = None
    for raw in text.splitlines():
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
        rules = [
            _Rule(allow, len(path), _compile(path))
            for group in chosen
            for allow, path in group
        ]
        return cls(rules)

    def allowed(self, path: str) -> bool:
        """Whether ``path`` (URL path plus any ``?query``) may be fetched."""
        path = unquote(path or "/")
        if path == "/robots.txt":
            return True
        best: _Rule | None = None
        for rule in self._rules:
            if rule.pattern.match(path) and (
                best is None
                or rule.length > best.length
                or (rule.length == best.length and rule.allow)
            ):
                best = rule
        return self._default if best is None else best.allow
