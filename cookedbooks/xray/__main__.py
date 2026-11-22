"""`python -m cookedbooks.xray EPISODE_DIR [--issue ID | --all] [--color auto|always|never]`.

Without --issue or --all: one line per issue and look-alike counts by kind.
Exit 0 on success; 2 with one short line on stderr for an unreadable folder,
a missing truth.json or an unknown issue ID (bad usage also exits 2).
Read-only: nothing is written into the episode folder.
"""

from __future__ import annotations

import argparse
import sys

from cookedbooks.xray.case import EpisodeView, XrayError, build_case
from cookedbooks.xray.render import render_case, render_list


def _fail(message: str) -> int:
    print("xray: " + message, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cookedbooks.xray",
                                     description="Case file of injected errors in one episode (jury-side, read-only).")
    parser.add_argument("episode_dir", help="episode folder")
    which = parser.add_mutually_exclusive_group()
    which.add_argument("--issue", help="issue_id from truth.json")
    which.add_argument("--all", action="store_true", help="every case file, in truth order")
    parser.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    args = parser.parse_args(argv)
    color = args.color == "always" or (args.color == "auto" and sys.stdout.isatty())
    try:
        view = EpisodeView(args.episode_dir)
        if args.issue is not None:
            position = view.find_issue(args.issue)
            if position is None:
                return _fail("в truth.json нет ошибки с таким issue_id")
            text = render_case(view, build_case(view, position), color=color)
        elif args.all:
            if not view.issues:
                text = render_list(view, color=color)
            else:
                text = "\n".join(render_case(view, build_case(view, p), color=color)
                                 for p in range(len(view.issues)))
        else:
            text = render_list(view, color=color)
    except XrayError as exc:
        return _fail(str(exc))
    except Exception as exc:  # a display tool never shows a traceback
        return _fail(f"не удалось показать эпизод ({type(exc).__name__})")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
