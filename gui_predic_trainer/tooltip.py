# ============================================================
# File: tooltip.py
# Author: Yael Vicente
# Date: 2025-May-06
# Description:
#   Provides a reusable tooltip class for displaying contextual
#   descriptions when hovering over GUI elements in a Tkinter interface.
# ============================================================

import tkinter as tk

class ToolTip:
    """
    A simple tooltip class for displaying help text when hovering over widgets.
    """

    def __init__(self, widget, text: str):
        """
        Initialize the tooltip and bind events to the target widget.

        Args:
            widget (tk.Widget): The widget to attach the tooltip to.
            text (str): The tooltip message to display.
        """
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self.enter)
        self.widget.bind("<Leave>", self.leave)

    def enter(self, event=None):
        """Display the tooltip when the cursor enters the widget area."""
        if self.tipwindow:
            return

        x, y, _, _ = self.widget.bbox("insert")
        x = x + self.widget.winfo_rootx() + 25
        y = y + self.widget.winfo_rooty() + 20

        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")

        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=300
        )
        label.pack(ipadx=1)

    def leave(self, event=None):
        """Destroy the tooltip when the cursor leaves the widget area."""
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None
