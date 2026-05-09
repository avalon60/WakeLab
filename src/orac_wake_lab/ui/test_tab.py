"""Phase 1 test placeholder tab for Orac Wake Lab."""
# Author: Clive Bostock
# Date: 2026-05-09
# Description: Documents that live testing belongs to Phase 2.

from __future__ import annotations

import customtkinter as ctk


class TestTab(ctk.CTkFrame):
    """Placeholder test tab."""

    def __init__(self, master: ctk.CTkBaseClass, app: object) -> None:
        """Create the Phase 1 test placeholder."""
        super().__init__(master)
        del app
        ctk.CTkLabel(
            self,
            text=(
                "Live microphone testing and threshold tuning are planned "
                "for Phase 2."
            ),
        ).pack(anchor="w", padx=12, pady=12)
