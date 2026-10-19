from pathlib import Path

import pytest

from ragsearch.config import ConfigError, Settings, load_settings, read_dotenv


def test_defaults_suit_small_local_models():
    settings = load_settings({})
    assert settings.ollama_host == "http://127.0.0.1:11434"
    assert settings.embed_model == "nomic-embed-text"
    assert settings.gen_model == "llama3.2:3b"
    assert settings.data_dir == Path("data")
    assert settings.tavily_api_key is None


def test_environment_variables_override_defaults():
    settings = load_settings(
        {
            "OLLAMA_HOST": "http://gpu-box:11434",
            "EMBED_MODEL": "mxbai-embed-large",
            "GEN_MODEL": "qwen2.5:3b",
            "DATA_DIR": "elsewhere",
            "TAVILY_API_KEY": "tvly-test",
        }
    )
    assert settings.ollama_host == "http://gpu-box:11434"
    assert settings.embed_model == "mxbai-embed-large"
    assert settings.gen_model == "qwen2.5:3b"
    assert settings.data_dir == Path("elsewhere")
    assert settings.tavily_api_key == "tvly-test"


def test_ollama_host_without_scheme_gets_http_prefix():
    # Ollama's own OLLAMA_HOST is often written as "127.0.0.1:11434".
    assert load_settings({"OLLAMA_HOST": "127.0.0.1:11434"}).ollama_host == (
        "http://127.0.0.1:11434"
    )


def test_empty_api_key_is_treated_as_missing():
    assert load_settings({"TAVILY_API_KEY": ""}).tavily_api_key is None


def test_api_key_never_appears_in_repr():
    settings = Settings(tavily_api_key="tvly-secret-value")
    assert "tvly-secret-value" not in repr(settings)


def test_index_backend_defaults_to_numpy_and_can_be_faiss():
    assert load_settings({}).index_backend == "numpy"
    assert load_settings({"INDEX_BACKEND": "FAISS"}).index_backend == "faiss"


def test_unknown_index_backend_is_rejected():
    with pytest.raises(ValueError, match="INDEX_BACKEND"):
        load_settings({"INDEX_BACKEND": "annoy"})


def test_web_settings_have_safe_defaults_and_can_be_overridden():
    defaults = load_settings({})
    assert defaults.tavily_monthly_cap == 800
    assert defaults.web_fallback_threshold == 0.55
    custom = load_settings({"TAVILY_MONTHLY_CAP": "100", "WEB_FALLBACK_THRESHOLD": "0.3"})
    assert custom.tavily_monthly_cap == 100
    assert custom.web_fallback_threshold == 0.3


def test_embedder_defaults_to_ollama_and_can_be_hash():
    assert load_settings({}).embedder == "ollama"
    assert load_settings({"EMBEDDER": "hash"}).embedder == "hash"
    with pytest.raises(ValueError, match="EMBEDDER"):
        load_settings({"EMBEDDER": "word2vec"})


def test_read_dotenv_parses_simple_key_value_lines(tmp_path):
    path = tmp_path / ".env"
    path.write_text(
        "# comment\n\nTAVILY_API_KEY=tvly-abc\nGEN_MODEL = 'qwen2.5:3b'\n"
        'DATA_DIR="my data"\nEMPTY=\nnot a setting\n',
        encoding="utf-8",
    )
    assert read_dotenv(path) == {
        "TAVILY_API_KEY": "tvly-abc",
        "GEN_MODEL": "qwen2.5:3b",
        "DATA_DIR": "my data",
    }


def test_read_dotenv_missing_file_is_empty(tmp_path):
    assert read_dotenv(tmp_path / ".env") == {}


def test_load_settings_reads_dotenv_in_current_directory(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TAVILY_API_KEY=tvly-from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert load_settings().tavily_api_key == "tvly-from-file"


def test_real_environment_variables_override_dotenv(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("TAVILY_API_KEY=tvly-from-file\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-from-shell")
    assert load_settings().tavily_api_key == "tvly-from-shell"


# --- clear errors for bad values, tolerant .env parsing ---------------------------


@pytest.mark.parametrize(
    "name,value",
    [("TAVILY_MONTHLY_CAP", "abc"), ("TAVILY_MONTHLY_CAP", "800.5"),
     ("TAVILY_MONTHLY_CAP", "-1"), ("WEB_FALLBACK_THRESHOLD", "0,5"),
     ("WEB_FALLBACK_THRESHOLD", "nan"), ("EMBEDDER", "word2vec"),
     ("INDEX_BACKEND", "annoy")],
)
def test_bad_values_raise_a_config_error_naming_the_variable(name, value):
    with pytest.raises(ConfigError, match=name):
        load_settings({name: value})


def test_empty_values_fall_back_to_defaults():
    settings = load_settings({"TAVILY_MONTHLY_CAP": "", "WEB_FALLBACK_THRESHOLD": "  ",
                              "EMBEDDER": "", "GEN_MODEL": ""})
    assert settings.tavily_monthly_cap == 800
    assert settings.web_fallback_threshold == 0.55
    assert settings.embedder == "ollama"
    assert settings.gen_model == "llama3.2:3b"


def test_values_are_stripped():
    settings = load_settings({"EMBEDDER": "hash ", "TAVILY_API_KEY": " tvly-abc\n",
                              "TAVILY_MONTHLY_CAP": " 100 "})
    assert settings.embedder == "hash"
    assert settings.tavily_api_key == "tvly-abc"
    assert settings.tavily_monthly_cap == 100


def test_config_error_is_a_value_error():
    assert issubclass(ConfigError, ValueError)


def test_dotenv_written_by_powershell_as_utf16_is_read(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes("TAVILY_API_KEY=tvly-abc\r\n".encode("utf-16"))  # BOM + UTF-16 LE
    assert read_dotenv(path) == {"TAVILY_API_KEY": "tvly-abc"}


def test_dotenv_inline_comments_are_stripped_unless_quoted(tmp_path):
    path = tmp_path / ".env"
    path.write_text("TAVILY_MONTHLY_CAP=800  # spare 200\nGEN_MODEL='a # b'\n",
                    encoding="utf-8")
    assert read_dotenv(path) == {"TAVILY_MONTHLY_CAP": "800", "GEN_MODEL": "a # b"}


def test_unreadable_dotenv_is_a_config_error(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes(b"KEY=\xff\xfe\xfa\x80 bad")
    with pytest.raises(ConfigError, match=".env"):
        read_dotenv(path)


@pytest.mark.parametrize(
    "value,expected",
    [("http://localhost:11434", "http://127.0.0.1:11434"),
     ("localhost:11434", "http://127.0.0.1:11434"),
     ("http://LOCALHOST:8080", "http://127.0.0.1:8080"),
     ("http://gpu-box:11434", "http://gpu-box:11434")],
)
def test_localhost_is_rewritten_to_ipv4(value, expected):
    # On Windows "localhost" tries IPv6 (::1) first; Ollama listens on IPv4 only,
    # which cost ~5 s per request before falling back.
    assert load_settings({"OLLAMA_HOST": value}).ollama_host == expected
