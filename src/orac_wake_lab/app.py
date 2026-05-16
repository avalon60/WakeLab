"""Command-line entry point for WakeLab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Launches the WakeLab CustomTkinter application.

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    """Build the application argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser.
    """
    return argparse.ArgumentParser(
        prog="wakelab",
        description="Launch the WakeLab wake-word training utility.",
    )


def main() -> int:
    """Run the WakeLab application.

    Returns:
        int: Process exit code.
    """
    parser = build_parser()
    parser.parse_args()

    try:
        from orac_wake_lab.ui.main_window import OracWakeLabApp
    except ImportError as exc:
        raise SystemExit(
            "Unable to import WakeLab UI: "
            f"{type(exc).__name__}: {exc}\n"
            f"Interpreter: {sys.executable}"
        ) from exc

    try:
        app = OracWakeLabApp()
        app.mainloop()
    except Exception as exc:
        raise SystemExit(
            "WakeLab failed to start: "
            f"{type(exc).__name__}: {exc}\n"
            f"Interpreter: {sys.executable}"
        ) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
