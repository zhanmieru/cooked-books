"""`python -m cookedbooks.verifier <episode_dir> [more ...] [--json]`.

Exit status 0 only if every episode is ok, 1 otherwise, 2 for bad usage.
"""

from __future__ import annotations

import argparse
import json
import sys

from .pipeline import verify_episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cookedbooks.verifier",
                                     description="Independently verify generated CookedBooks episodes.")
    parser.add_argument("episode_dirs", nargs="+", help="episode folders to check")
    parser.add_argument("--json", action="store_true", help="print one JSON report per line")
    args = parser.parse_args(argv)
    all_ok = True
    for directory in args.episode_dirs:
        report = verify_episode(directory)
        all_ok = all_ok and report.ok
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True, default=str))
        else:
            print(report.render())
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
