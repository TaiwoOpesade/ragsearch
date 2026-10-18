import subprocess
import sys
from pathlib import Path

import pytest

from ragsearch.cli import main
from ragsearch.config import Settings

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", ollama_host="http://127.0.0.1:1")


def run(argv, settings, capsys):
    code = main(argv, settings=settings)
    out, err = capsys.readouterr()
    return code, out, err


def test_ingest_then_search_returns_the_right_fixture_first(settings, capsys):
    code, out, _ = run(["ingest", str(FIXTURES), "--embedder", "hash"], settings, capsys)
    assert code == 0
    assert "4 sources" in out

    code, out, _ = run(
        ["search", "which structure is last in first out", "-k", "3", "--embedder", "hash"],
        settings,
        capsys,
    )
    assert code == 0
    first = out.strip().splitlines()[0]
    assert first.startswith("1.")
    assert first.endswith("stacks.md")
    assert "last in first out" in out  # snippet is shown


def test_search_prints_score_with_three_decimals(settings, capsys):
    run(["ingest", str(FIXTURES), "--embedder", "hash"], settings, capsys)
    _, out, _ = run(["search", "glucose", "-k", "1", "--embedder", "hash"], settings, capsys)
    score = out.split()[1]
    assert len(score.split(".")[1]) == 3


def test_reingest_reports_zero_new_chunks(settings, capsys):
    run(["ingest", str(FIXTURES), "--embedder", "hash"], settings, capsys)
    _, out, _ = run(["ingest", str(FIXTURES), "--embedder", "hash"], settings, capsys)
    assert "0 new chunks" in out
    assert "4 unchanged" in out


def test_search_with_empty_store_says_so(settings, capsys):
    code, out, _ = run(["search", "anything", "--embedder", "hash"], settings, capsys)
    assert code == 0
    assert "No results" in out


def test_data_dir_flag_overrides_settings(tmp_path, settings, capsys):
    other = tmp_path / "other"
    run(["ingest", str(FIXTURES), "--embedder", "hash", "--data-dir", str(other)],
        settings, capsys)
    assert (other / "manifest.json").exists()
    assert not settings.data_dir.exists()


def test_unreachable_ollama_gives_clear_error_and_exit_code_1(settings, capsys):
    code, _, err = run(["search", "anything", "--embedder", "ollama"], settings, capsys)
    assert code == 1
    assert "Ollama" in err


def test_unknown_embedder_is_rejected(settings, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["search", "q", "--embedder", "word2vec"], settings=settings)
    assert exc.value.code == 2


def test_python_dash_m_entry_point_shows_help():
    result = subprocess.run(
        [sys.executable, "-m", "ragsearch", "--help"],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent,
        check=False,
    )
    assert result.returncode == 0
    assert "ingest" in result.stdout and "search" in result.stdout


def test_cli_works_with_the_faiss_backend(tmp_path, capsys):
    settings = Settings(data_dir=tmp_path / "data", index_backend="faiss")
    run(["ingest", str(FIXTURES), "--embedder", "hash"], settings, capsys)
    _, out, _ = run(["search", "chlorophyll glucose", "-k", "1", "--embedder", "hash"],
                    settings, capsys)
    assert out.splitlines()[0].endswith("photosynthesis.md")


def test_embedder_setting_is_used_when_flag_is_omitted(tmp_path, capsys):
    settings = Settings(data_dir=tmp_path / "data", embedder="hash")
    code, out, _ = run(["ingest", str(FIXTURES)], settings, capsys)
    assert code == 0
    assert "4 sources" in out
