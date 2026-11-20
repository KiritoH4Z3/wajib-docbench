"""python -m data.generator [--only DOC_ID ...] [--skip-render] [--check] [--out DIR]

Without --check: writes data/ground_truth/<id>.json for all 85 documents and renders
data/docs/<id>.pdf (clean and poisoned) and <id>.jpg (scanned).
With --check: validates what is on disk and exits 1 on any failure. --skip-render with
--check limits it to the ground truth (no PDF, JPEG or browser).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from bench.paths import DATA
from data.generator.plan import all_doc_ids


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m data.generator",
        description="Generate the synthetic UAE document set, or check it.",
    )
    parser.add_argument(
        "--only", action="append", metavar="DOC_ID", help="generate only this document (repeatable)"
    )
    parser.add_argument(
        "--skip-render", action="store_true", help="ground truth only: no PDF, JPEG or browser"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate the generated set and exit non-zero on any failure",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DATA,
        help="output root holding docs/ and ground_truth/ (default: data)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.check:
        from data.generator.check import run_checks

        report = run_checks(args.out, render_checks=not args.skip_render)
        print(report.summary())
        return 0 if report.ok else 1

    unknown = sorted(set(args.only or []) - set(all_doc_ids()))
    if unknown:
        print(
            f"Unknown document id(s): {', '.join(unknown)}. "
            "Ids look like tl-ar-01, inv-en-02.poison or qt-bi-05.scan.",
            file=sys.stderr,
        )
        return 2
    from data.generator.build import generate

    print(generate(args.out, only=args.only, skip_render=args.skip_render))
    return 0


if __name__ == "__main__":
    sys.exit(main())
