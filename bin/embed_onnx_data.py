#!/usr/bin/env python3
"""Embed ONNX external data into a single self-contained ONNX model file.

Author: Clive Bostock
Date: 15-May-2026
Purpose: Convert an ONNX model with external data into a single embedded ONNX file.
Usage: ./bin/embed_onnx_data.py --source-onnx model.onnx --target-onnx out/model.onnx
"""

# Author: Clive Bostock
# Date: 15-May-2026
# Description: Convert an ONNX model with external data into a single embedded ONNX file.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orac_wake_lab.services.orac_export import embed_onnx_external_data_file


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Embed ONNX external .data sidecar contents into one .onnx file."
    )
    parser.add_argument(
        "--source-onnx",
        required=True,
        type=Path,
        help="Source .onnx file that currently references external data.",
    )
    parser.add_argument(
        "--target-onnx",
        required=True,
        type=Path,
        help="Target self-contained .onnx file to create.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point.

    Returns:
        int: Process exit status code.
    """
    args = parse_args()

    try:
        embed_onnx_external_data_file(
            source_onnx=args.source_onnx.expanduser().resolve(),
            target_onnx=args.target_onnx.expanduser().resolve(),
        )
        print(f"Source ONNX  : {args.source_onnx.expanduser().resolve()}")
        print(f"Target ONNX  : {args.target_onnx.expanduser().resolve()}")
        print("External data: embedded")
        print("Verification : OK")
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
