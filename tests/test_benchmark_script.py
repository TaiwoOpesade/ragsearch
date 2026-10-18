import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "benchmark_index.py"


def test_benchmark_script_writes_a_markdown_table_of_measurements(tmp_path):
    out = tmp_path / "bench.md"
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--sizes", "200", "500", "--dim", "16",
         "--queries", "5", "--output", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    text = out.read_text(encoding="utf-8")
    assert "| Vectors |" in text
    rows = [line for line in text.splitlines() if line.startswith(("| 200 ", "| 500 "))]
    assert len(rows) == 2
    # every numeric cell is a real measured, positive number
    for row in rows:
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert all(float(c.rstrip("x")) > 0 for c in cells[1:])
