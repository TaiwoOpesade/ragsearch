"""RFC 9309 robots.txt matching, checked rule by rule."""

import pytest

from ragsearch.web.robots import RobotsRules

AGENT = "ragsearch"


def rules(text):
    return RobotsRules.parse(text, AGENT)


def test_longest_match_wins_not_first_match():
    r = rules("User-agent: *\nAllow: /\nDisallow: /private\n")
    assert r.allowed("/public")
    assert not r.allowed("/private/secret")


def test_allow_wins_a_tie_of_equal_length():
    r = rules("User-agent: *\nDisallow: /page\nAllow: /page\n")
    assert r.allowed("/page")


def test_more_specific_allow_reopens_a_disallowed_folder():
    r = rules("User-agent: *\nDisallow: /docs/\nAllow: /docs/public/\n")
    assert not r.allowed("/docs/internal")
    assert r.allowed("/docs/public/intro")


@pytest.mark.parametrize(
    "path,allowed",
    [("/data/x.json", False), ("/data/x.json?v=1", True), ("/data/x.html", True)],
)
def test_dollar_anchors_the_end_of_the_path(path, allowed):
    assert rules("User-agent: *\nDisallow: /*.json$\n").allowed(path) is allowed


@pytest.mark.parametrize(
    "path,allowed", [("/search", False), ("/search?q=1", False), ("/searching", False),
                     ("/a/search", True)],
)
def test_star_matches_any_sequence(path, allowed):
    assert rules("User-agent: *\nDisallow: /search*\n").allowed(path) is allowed


def test_specific_user_agent_group_replaces_the_star_group():
    text = "User-agent: *\nDisallow: /\n\nUser-agent: RagSearch\nDisallow: /admin\n"
    r = rules(text)
    assert r.allowed("/docs")  # our own group applies, not "*"
    assert not r.allowed("/admin")


def test_groups_for_the_same_agent_are_merged():
    text = "User-agent: *\nDisallow: /a\n\nUser-agent: *\nDisallow: /b\n"
    r = rules(text)
    assert not r.allowed("/a") and not r.allowed("/b")


def test_several_user_agent_lines_share_one_group():
    text = "User-agent: googlebot\nUser-agent: ragsearch\nDisallow: /x\n"
    assert not rules(text).allowed("/x")


def test_other_agents_rules_do_not_apply_to_us():
    assert rules("User-agent: googlebot\nDisallow: /\n").allowed("/anything")


def test_empty_disallow_means_allow_everything():
    assert rules("User-agent: *\nDisallow:\n").allowed("/anything")


def test_comments_case_and_whitespace_are_ignored():
    text = "# site rules\nUSER-AGENT : *   # everyone\n  disallow : /tmp  # scratch\n"
    assert not rules(text).allowed("/tmp/file")


def test_robots_txt_itself_is_always_allowed():
    assert rules("User-agent: *\nDisallow: /\n").allowed("/robots.txt")


def test_percent_encoding_is_compared_consistently():
    r = rules("User-agent: *\nDisallow: /caf%C3%A9\n")
    assert not r.allowed("/caf%c3%a9/menu")
    assert not r.allowed("/café/menu")


def test_allow_all_and_disallow_all():
    assert RobotsRules.allow_all().allowed("/x")
    assert not RobotsRules.disallow_all().allowed("/x")
