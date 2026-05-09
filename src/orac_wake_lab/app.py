"""Command-line entry point for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Launches the Orac Wake Lab CustomTkinter application.

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the application argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    return argparse.ArgumentParser(
        prog="orac-wake-lab",
        description="Launch the Orac Wake Lab wake-word training utility.",
    )


def main() -> int:
    """Run the Orac Wake Lab application.

    Returns:
        int: Process exit code.
    """
    parser = build_parser()
    parser.parse_args()

    try:
        from orac_wake_lab.ui.main_window import OracWakeLabApp
    except ImportError as exc:
        raise SystemExit(
            "Unable to import Orac Wake Lab UI: "
            f"{type(exc).__name__}: {exc}\n"
            f"Interpreter: {sys.executable}"
        ) from exc

    try:
        app = OracWakeLabApp()
        app.mainloop()
    except Exception as exc:
        raise SystemExit(
            "Orac Wake Lab failed to start: "
            f"{type(exc).__name__}: {exc}\n"
            f"Interpreter: {sys.executable}"
        ) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
