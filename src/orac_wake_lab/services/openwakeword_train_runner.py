"""Run openWakeWord training with compatibility fixes."""
# Author: Clive Bostock
# Date: 2026-05-13
# Description: Provides a safe wrapper around the openWakeWord train module.

from __future__ import annotations

import argparse
import runpy
import warnings


def _patch_store_true_defaults() -> None:
    """Patch invalid string defaults on upstream ``store_true`` arguments."""
    original_add_argument = argparse.ArgumentParser.add_argument

    def add_argument_with_boolean_defaults(
        self: argparse.ArgumentParser,
        *args: object,
        **kwargs: object,
    ) -> argparse.Action:
        if (
            kwargs.get("action") == "store_true"
            and kwargs.get("default") == "False"
        ):
            kwargs["default"] = False
        return original_add_argument(self, *args, **kwargs)

    argparse.ArgumentParser.add_argument = add_argument_with_boolean_defaults


def _filter_noisy_third_party_warnings() -> None:
    """Suppress known third-party deprecation warnings in training logs."""
    filters = [
        (
            UserWarning,
            r"pkg_resources is deprecated as an API",
            r"pronouncing\.__init__",
        ),
        (
            FutureWarning,
            r"Transforms now expect an `output_type` argument",
            r"torch_audiomentations\.",
        ),
        (
            FutureWarning,
            r"Transforms now expect an `output_type` argument",
            r"audiomentations\.",
        ),
        (
            UserWarning,
            r"Warning: input samples dtype is np\.float64",
            r"audiomentations\.",
        ),
    ]
    for category, message, module in filters:
        warnings.filterwarnings(
            "ignore",
            message=message,
            category=category,
            module=module,
        )


def main() -> None:
    """Run ``openwakeword.train`` as ``__main__`` after applying patches."""
    _filter_noisy_third_party_warnings()
    _patch_store_true_defaults()
    runpy.run_module("openwakeword.train", run_name="__main__")


if __name__ == "__main__":
    main()
