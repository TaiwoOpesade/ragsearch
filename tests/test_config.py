from pathlib import Path

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
