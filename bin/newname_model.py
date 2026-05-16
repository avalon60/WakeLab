#!/usr/bin/env python3
"""Repackage an ONNX model and rewrite its external data filename.

Author: Clive Bostock
Date: 15-May-2026
Purpose: Repackage an ONNX model that uses an external .data sidecar file.
Usage: ./bin/newname_model.py --source-onnx model.onnx --target-onnx out/model.onnx --target-data out/model.data
"""

# Author: Clive Bostock
# Date: 15-May-2026
# Description: Repackage an ONNX model that uses an external .data sidecar file.

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import onnx


def get_external_locations(model: onnx.ModelProto) -> set[str]:
    """Return external data locations referenced by an ONNX model.

    Args:
        model (onnx.ModelProto): ONNX model loaded with external data disabled.

    Returns:
        set[str]: External data file locations referenced by model tensors.
    """
    locations: set[str] = set()

    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                locations.add(entry.value)

    return locations


def rewrite_external_location(model: onnx.ModelProto, new_location: str) -> int:
    """Rewrite ONNX external data locations.

    Args:
        model (onnx.ModelProto): ONNX model loaded with external data disabled.
        new_location (str): New external data filename to store in the model.

    Returns:
        int: Number of rewritten location entries.
    """
    rewrite_count = 0

    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = new_location
                rewrite_count += 1

    return rewrite_count


def repackage_model(
    source_onnx: Path,
    target_onnx: Path,
    target_data: Path,
) -> None:
    """Copy and repackage an ONNX model with a renamed external data file.

    Args:
        source_onnx (Path): Source ONNX model path.
        target_onnx (Path): Target ONNX model path.
        target_data (Path): Target external data file path.

    Raises:
        FileNotFoundError: If the source ONNX or external data file is missing.
        ValueError: If the model has no external data or multiple sidecar files.
    """
    if not source_onnx.exists():
        raise FileNotFoundError(f"Source ONNX file not found: {source_onnx}")

    model = onnx.load(str(source_onnx), load_external_data=False)
    locations = get_external_locations(model)

    if not locations:
        raise ValueError(f"No external data references found in: {source_onnx}")

    if len(locations) > 1:
        joined_locations = ", ".join(sorted(locations))
        raise ValueError(
            "This script expects one external data file only. "
            f"Found: {joined_locations}"
        )

    source_location = next(iter(locations))
    source_data = source_onnx.parent / source_location

    if not source_data.exists():
        raise FileNotFoundError(
            f"External data file referenced by model was not found: {source_data}"
        )

    target_onnx.parent.mkdir(parents=True, exist_ok=True)
    target_data.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy2(source_data, target_data)

    rewrite_count = rewrite_external_location(model, target_data.name)
    onnx.save_model(model, str(target_onnx))

    # Reload with external data enabled to confirm the new sidecar path works.
    onnx.load(str(target_onnx), load_external_data=True)

    print(f"Source ONNX       : {source_onnx}")
    print(f"Source data       : {source_data}")
    print(f"Target ONNX       : {target_onnx}")
    print(f"Target data       : {target_data}")
    print(f"Rewritten entries : {rewrite_count}")
    print("Verification      : OK")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Repackage an ONNX model that uses one external .data sidecar file."
        )
    )
    parser.add_argument(
        "--source-onnx",
        required=True,
        type=Path,
        help="Source .onnx file.",
    )
    parser.add_argument(
        "--target-onnx",
        required=True,
        type=Path,
        help="Target .onnx file to create.",
    )
    parser.add_argument(
        "--target-data",
        required=True,
        type=Path,
        help="Target .data sidecar file to create.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line entry point.

    Returns:
        int: Process exit status code.
    """
    args = parse_args()

    try:
        repackage_model(
            source_onnx=args.source_onnx.expanduser().resolve(),
            target_onnx=args.target_onnx.expanduser().resolve(),
            target_data=args.target_data.expanduser().resolve(),
        )
    except (
        FileNotFoundError,
        OSError,
        ValueError,
        onnx.checker.ValidationError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
