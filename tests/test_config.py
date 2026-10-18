from pathlib import Path

import pytest

from ragsearch.config import Settings, load_settings


def test_defaults_suit_small_local_models():
    settings = load_settings({})
    assert settings.ollama_host == "http://localhost:11434"
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
    assert defaults.web_fallback_threshold == 0.5
    custom = load_settings({"TAVILY_MONTHLY_CAP": "100", "WEB_FALLBACK_THRESHOLD": "0.3"})
    assert custom.tavily_monthly_cap == 100
    assert custom.web_fallback_threshold == 0.3
