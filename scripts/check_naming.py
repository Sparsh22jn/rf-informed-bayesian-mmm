"""Fail if any tracked file contains an old-codebase naming pattern.

Patterns describe the *shape* of client/legacy identifiers (see
naming_patterns.txt) rather than listing the identifiers themselves, so the
pattern file is safe to commit and this check is safe to run in CI.
"""
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PATTERNS_FILE = REPO_ROOT / "scripts" / "naming_patterns.txt"

# Documentation *about* the check necessarily quotes the pattern shapes as
# examples (see TASKS.md task 0.2) — that's not a leak, exempt it.
EXEMPT_FILES = {PATTERNS_FILE.resolve(), (REPO_ROOT / "TASKS.md").resolve()}


def load_patterns() -> list[re.Pattern]:
    patterns = []
    for line in PATTERNS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        patterns.append(re.compile(line))
    return patterns


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return [REPO_ROOT / line for line in result.stdout.splitlines()]


def find_hits(patterns: list[re.Pattern]) -> list[str]:
    hits = []
    for path in tracked_files():
        if path.resolve() in EXEMPT_FILES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            for pattern in patterns:
                if pattern.search(line):
                    rel = path.relative_to(REPO_ROOT)
                    hits.append(f"{rel}:{lineno}: matches /{pattern.pattern}/ -> {line.strip()}")
    return hits


def main() -> int:
    patterns = load_patterns()
    hits = find_hits(patterns)

    if hits:
        print("Old-codebase naming patterns found in tracked files:")
        for hit in hits:
            print(f"  {hit}")
        return 1

    print(f"OK — {len(patterns)} patterns checked, 0 matches in tracked tree.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
