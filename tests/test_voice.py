"""Copy gate: no em dashes and no machine-sounding vocabulary in anything a reader sees.

Silent failure this catches: a banned word or an em dash slipping into the README, an ADR,
a docstring that becomes an error message, or a document template, with every other test
green. Strings are read through Python's own parser, so comments are not text and a quote
inside a string cannot hide the next literal. The gate proves it can fail before it is
trusted to pass.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EM_DASH = chr(0x2014)
EN_DASH = chr(0x2013)

BANNED = [
    "unlock", "elevate", "empower", "leverage", "seamless", "seamlessly",
    "effortless", "effortlessly", "robust", "cutting-edge", "state-of-the-art",
    "best-in-class", "world-class", "unparalleled", "game-changer", "revolutionize",
    "transform the way", "supercharge", "streamline", "next-level", "holistic", "synergy",
    "delve", "navigate the complexities", "fast-paced", "ever-evolving", "journey",
    "curated", "crafted with care", "designed with you in mind", "tailored to your needs",
    "at your fingertips", "one-stop", "look no further", "say goodbye to",
    "welcome to the future", "dive in", "a must-have", "peace of mind", "hassle-free",
    "no-brainer", "ai-powered",
]
BANNED_RE = re.compile(
    r"(?<![a-z])(" + "|".join(re.escape(w) for w in BANNED) + r")(?![a-z])", re.IGNORECASE
)
# "Harness" is banned as a verb ("harness the power of"), not as the engineering noun
# ("the test harness logs every request").
HARNESS_VERB = re.compile(
    r"(?<![a-z])harness(es|ed|ing)?\s+(the|its|their|our|your|this|that|ai|power)(?![a-z])",
    re.IGNORECASE,
)
# A spaced en dash or a spaced double hyphen doing an em dash's job. Ranges between
# digits ("2019–2024", "9:00 – 18:00") are fine.
SPACED_DASH = re.compile(rf"(?<!\d)\s({EN_DASH}|--)\s(?!\d)")

# path suffix -> reason. Keep this list short and justified.
ALLOW: dict[str, str] = {}

TEXT_GLOBS = ["*.md", "docs/**/*.md", "data/generator/templates/**/*", "config/*.yaml"]
PY_DIRS = ["bench", "extract", "ocr", "guardrails", "eval", "perf", "data/generator", "tests"]


def violations(text: str) -> list[str]:
    found = []
    if EM_DASH in text:
        found.append("em dash")
    for m in BANNED_RE.finditer(text):
        found.append(f"banned word {m.group(0)!r}")
    if HARNESS_VERB.search(text):
        found.append("harness used as a verb")
    if SPACED_DASH.search(text):
        found.append("spaced dash used as an em dash")
    return found


def python_strings(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def text_files() -> list[Path]:
    files: set[Path] = set()
    for pattern in TEXT_GLOBS:
        files.update(p for p in ROOT.glob(pattern) if p.is_file())
    return sorted(files)


def python_files() -> list[Path]:
    return sorted(
        p
        for d in PY_DIRS
        for p in (ROOT / d).rglob("*.py")
        if p.name != "test_voice.py"
    )


def test_the_gate_catches_planted_violations(tmp_path):
    planted = {
        "em": f"Fast{EM_DASH}and cheap",
        "word": "A seamless pipeline",
        "word_caps": "ROBUST results",
        "harness_verb": "We harness the power of models",
        "spaced": f"OCR {EN_DASH} then the model",
    }
    for text in planted.values():
        assert violations(text), text
    src = tmp_path / "planted.py"
    src.write_text(f'X = "ok"\nY = f"{{X}} is a game-changer"\nZ = """{EM_DASH}"""\n')
    hits = [v for s in python_strings(src) for v in violations(s)]
    assert len(hits) == 2


def test_the_gate_allows_ranges_and_hyphens():
    assert not violations(f"Valid 2024{EN_DASH}2026, open 9:00 {EN_DASH} 18:00")
    assert not violations("A well-known, multi-page, right-to-left layout")
    assert not violations("uv run tesseract --psm 6")
    assert not violations("The harness logs TTFT for every request")


def test_the_tree_is_not_empty():
    assert len(text_files()) >= 3
    assert len(python_files()) >= 5


@pytest.mark.parametrize("path", text_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_text_files(path):
    rel = str(path.relative_to(ROOT))
    if rel in ALLOW:
        return
    assert not violations(path.read_text(encoding="utf-8")), rel


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_python_strings(path):
    rel = str(path.relative_to(ROOT))
    hits = [(s[:60], v) for s in python_strings(path) for v in violations(s)]
    assert not hits, f"{rel}: {hits}"
