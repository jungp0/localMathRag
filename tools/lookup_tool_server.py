from __future__ import annotations

import sys

from lookup_tool.cli import main


def run() -> int:
    if len(sys.argv) > 1:
        return main(sys.argv[1:])
    return main(["serve", "--host", "127.0.0.1", "--port", "8765"])


if __name__ == "__main__":
    raise SystemExit(run())
