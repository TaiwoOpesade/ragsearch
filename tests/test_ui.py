"""The browser interface served by the API, and the protections that come with it."""

import json
import re
import warnings
from pathlib import Path

import pytest

from ragsearch.api import create_app
from ragsearch.config import Settings

with warnings.catch_warnings():
    # See tests/test_api.py: httpx2 is not an approved dependency.
    warnings.filterwarnings("ignore", message="Using `httpx` with `starlette.testclient`")
    from fastapi.testclient import TestClient

UI_DIR = Path(__file__).parent.parent / "ragsearch" / "ui"
UI_FILES = ["index.html", "app.css", "app.js"]


class NoGenerator:
    def generate(self, question, sources):
        raise AssertionError("not used")


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", embedder="hash")


def make_client(settings, **kwargs):
    app = create_app(settings=settings, generator=NoGenerator(), **kwargs)
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))


def test_root_serves_the_page(settings):
    response = make_client(settings).get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>ragsearch</title>" in response.text
    assert 'src="/ui/app.js"' in response.text
    assert 'href="/ui/app.css"' in response.text


def test_page_has_a_strict_content_security_policy(settings):
    csp = make_client(settings).get("/").headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "script-src 'self'" in csp
    assert "unsafe-inline" not in csp
    assert "frame-ancestors 'none'" in csp
    assert "connect-src 'self'" in csp


@pytest.mark.parametrize("name,kind", [("app.js", "javascript"), ("app.css", "text/css")])
def test_assets_are_served(settings, name, kind):
    response = make_client(settings).get(f"/ui/{name}")
    assert response.status_code == 200
    assert kind in response.headers["content-type"]


@pytest.mark.parametrize("name", UI_FILES)
def test_ui_loads_nothing_from_the_internet(name):
    text = (UI_DIR / name).read_text(encoding="utf-8")
    external = [
        r"""(src|href)\s*=\s*["']?(https?:)?//""",  # scripts, styles, images, fonts
        r"""url\(\s*["']?(https?:)?//""",           # CSS url()
        r"@import",
        r"""fetch\(\s*["'`]https?:""",
    ]
    for pattern in external:
        assert not re.search(pattern, text), f"{name} matches {pattern}"


def test_script_never_builds_html_from_strings():
    script = (UI_DIR / "app.js").read_text(encoding="utf-8")
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval("):
        assert sink not in script, f"app.js uses {sink}"


def test_inline_scripts_and_styles_are_not_used():
    page = (UI_DIR / "index.html").read_text(encoding="utf-8")
    assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", page)  # every script is a file
    assert "<style" not in page
    assert not re.search(r"\son[a-z]+\s*=", page)  # no onclick= style handlers
    assert "style=" not in page


def test_health_reports_web_search_status_and_credits(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    from datetime import date

    (data / "tavily_usage.json").write_text(
        json.dumps({"month": date.today().strftime("%Y-%m"), "credits": 4}), encoding="utf-8")
    with_key = Settings(data_dir=data, embedder="hash", tavily_api_key="tvly-test",
                        tavily_monthly_cap=800)
    body = make_client(with_key).get("/health").json()
    assert body["web_configured"] is True
    assert body["web_credits_used"] == 4
    assert body["web_credits_cap"] == 800

    without_key = make_client(Settings(data_dir=tmp_path / "other", embedder="hash"))
    assert without_key.get("/health").json()["web_configured"] is False


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "127.0.0.1"])
def test_local_host_headers_are_accepted(settings, host):
    client = make_client(settings)
    assert client.get("/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8000", "192.168.1.5:8000"])
def test_foreign_host_headers_are_refused(settings, host):
    # Blocks DNS rebinding: a web page whose domain is re-pointed at 127.0.0.1 must
    # not be able to drive this API (it can read local files via /ingest + /search).
    client = make_client(settings)
    assert client.get("/health", headers={"host": host}).status_code == 400
    assert client.get("/", headers={"host": host}).status_code == 400


def test_allowed_hosts_can_be_extended_for_lan_use(tmp_path):
    settings = Settings(data_dir=tmp_path / "data", embedder="hash",
                        allowed_hosts=("127.0.0.1", "localhost", "192.168.1.5"))
    client = make_client(settings)
    assert client.get("/health", headers={"host": "192.168.1.5:8000"}).status_code == 200


def test_allowed_hosts_setting_is_read_from_the_environment():
    from ragsearch.config import load_settings

    assert load_settings({}).allowed_hosts == ("127.0.0.1", "localhost")
    custom = load_settings({"ALLOWED_HOSTS": " 127.0.0.1, mylaptop.local "})
    assert custom.allowed_hosts == ("127.0.0.1", "mylaptop.local")


# --- third review ------------------------------------------------------------------


@pytest.mark.parametrize("caller", ["192.168.1.20", "10.0.0.5", "testclient"])
def test_callers_on_other_computers_are_refused_even_with_a_local_host_header(settings,
                                                                           caller):
    # A network client can send "Host: 127.0.0.1" itself, so the Host check alone does
    # not stop it; the API has no authentication and can read local files.
    app = create_app(settings=settings, generator=NoGenerator())
    remote = TestClient(app, base_url="http://127.0.0.1", client=(caller, 50000))
    response = remote.get("/health")
    assert response.status_code == 403
    assert "this computer" in response.text


@pytest.mark.parametrize("caller", ["127.0.0.1", "127.0.0.2", "::1"])
def test_callers_on_this_computer_are_served(settings, caller):
    app = create_app(settings=settings, generator=NoGenerator())
    local = TestClient(app, base_url="http://127.0.0.1", client=(caller, 50000))
    assert local.get("/health").status_code == 200


@pytest.mark.parametrize("path", ["/ui/index.html", "/ui/app.js", "/ui/app.css"])
def test_every_ui_file_gets_the_security_headers(settings, path):
    headers = make_client(settings).get(path).headers
    assert "frame-ancestors 'none'" in headers["content-security-policy"]
    assert headers["x-content-type-options"] == "nosniff"


def test_api_docs_keep_working_without_the_page_policy(settings):
    response = make_client(settings).get("/docs")
    assert response.status_code == 200
    assert "content-security-policy" not in response.headers  # /docs loads its own assets


@pytest.mark.parametrize("url", ["HTTPS://docs.example/", "Https://docs.example/a"])
def test_url_scheme_is_case_insensitive(url):
    from ragsearch.schemas import IngestRequest

    assert IngestRequest(url=url).url == url


@pytest.mark.parametrize("value", ["192.168.1.*", "::1", "a,,*"])
def test_unusable_allowed_hosts_are_a_clear_config_error(value):
    from ragsearch.config import ConfigError, load_settings

    with pytest.raises(ConfigError, match="ALLOWED_HOSTS"):
        load_settings({"ALLOWED_HOSTS": value})


def test_ipv6_and_leading_wildcards_are_accepted_in_allowed_hosts():
    from ragsearch.config import load_settings

    hosts = load_settings({"ALLOWED_HOSTS": "127.0.0.1,[::1],*.lan.example"}).allowed_hosts
    assert hosts == ("127.0.0.1", "[::1]", "*.lan.example")


def test_health_reports_the_web_fallback_threshold(settings):
    assert make_client(settings).get("/health").json()["web_fallback_threshold"] == 0.55


def test_stylesheet_never_hides_the_keyboard_focus_outline():
    css = (UI_DIR / "app.css").read_text(encoding="utf-8")
    assert "outline: none" not in css
