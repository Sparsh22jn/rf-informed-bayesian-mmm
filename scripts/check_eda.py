"""Re-executes notebooks/01_eda.ipynb headlessly via nbconvert, then asserts
the expected figures exist and the printed baseline share is in band.
Executing without error only proves it didn't crash -- this checks the
actual numbers, which is the point.

Usage: python scripts/check_eda.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "notebooks" / "01_eda.ipynb"
FIGS_DIR = REPO_ROOT / "docs" / "figs"

EXPECTED_FIGURES = (
    "eda_viewership_distribution.png",
    "eda_spend_alwayson.png",
    "eda_spend_eventtargeted.png",
    "eda_baseline_dominance.png",
)
BASELINE_SHARE_BAND = (0.60, 0.70)


def main() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--output",
            NOTEBOOK.name,
            str(NOTEBOOK),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, "notebook execution failed"

    for name in EXPECTED_FIGURES:
        path = FIGS_DIR / name
        assert path.exists(), f"expected figure missing: {path}"
        print(f"OK: {path} exists")

    import nbformat

    nb = nbformat.read(NOTEBOOK, as_version=4)
    stream_text = "".join(
        "".join(out.get("text", ""))
        for cell in nb.cells
        for out in cell.get("outputs", [])
        if out.get("output_type") == "stream"
    )
    match = re.search(r"Baseline share:\s*([\d.]+)%", stream_text)
    assert match, "could not find 'Baseline share: X%' in notebook output"
    baseline_share = float(match.group(1)) / 100.0

    low, high = BASELINE_SHARE_BAND
    print(f"\nPrinted baseline share: {baseline_share:.1%} (target {low:.0%}-{high:.0%})")
    assert low <= baseline_share <= high, (
        f"baseline share {baseline_share:.1%} outside target band {low:.0%}-{high:.0%}"
    )

    print("\nOK: all figures present, baseline share in band")


if __name__ == "__main__":
    main()
