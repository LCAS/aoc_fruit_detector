# ============================================================
# File: form_renderer.py
# Author: Yael Vicente
# Date: 2025-May-06
# Description:
#   Handles the dynamic creation and rendering of YAML configuration
#   forms within the GUI, including tooltips, field inputs, buttons,
#   directory selectors, and post-prediction summary display.
# ============================================================

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ruamel.yaml.comments import CommentedSeq
from constants import FIELD_DESCRIPTIONS
from tooltip import ToolTip

REQUIRED_FIELDS = {
    "aoc_fruit_detector.ros__parameters.min_depth",
    "aoc_fruit_detector.ros__parameters.max_depth",
    "aoc_fruit_detector.ros__parameters.constant_depth_value",
    "aoc_fruit_detector.ros__parameters.fruit_type",
    "aoc_fruit_detector.ros__parameters.pose3d_frame",
    "aoc_fruit_detector.ros__parameters.pose3d_tf",
    "aoc_fruit_detector.ros__parameters.verbose",
    "aoc_fruit_detector.ros__parameters.pub_verbose",
    "aoc_fruit_detector.ros__parameters.pub_markers",
    "aoc_fruit_detector.ros__parameters.use_ros",
    "datasets.train_dataset_name",
    "datasets.test_dataset_name",
    "datasets.validation_dataset_name",
    "files.config_file",
    "files.model_file",
    "files.config_file",
    "files.test_metadata_catalog_file",
    "files.train_dataset_catalog_file",
    "files.train_annotation_file",
    "files.test_annotation_file",
    "files.validation_annotation_file",
    "directories.train_image_dir",
    "directories.test_image_dir",
    "directories.validation_image_dir",
    "directories.training_output_dir",
    "directories.prediction_output_dir",
    "directories.prediction_json_dir",
    "training.epochs",
    "training.batch_size",
    "training.number_of_classes",
    "training.optimizer",
    "training.learning_rate",
    "settings.download_assets",
    "settings.rename_pred_images",
    "settings.segm_masks",
    "settings.bbox",
    "settings.show_orientation",
    "settings.fruit_type",
    "settings.validation_period",
    "settings.confidence_threshold",
    "settings.filename_patterns.rgb",
    "settings.filename_patterns.depth"
}

class FormRenderer:
    """
    Class responsible for rendering GUI components for YAML configuration.
    """

    def __init__(self, root):
        self.root = root
        self.entries = {}

    def clear_frame(self):
        """Remove all widgets from the main window."""
        for widget in self.root.winfo_children():
            widget.destroy()

    def render_title(self, text: str):
        """Display a section title."""
        tk.Label(self.root, text=text, font=("Arial", 14, "bold")).pack(pady=10)

    def render_button(self, text: str, command):
        """Display a button with the specified label and callback."""
        tk.Button(self.root, text=text, width=30, command=command).pack(pady=5)

    def render_form(self, yaml_data, save_callback, main_menu_callback):
        """
        Render the full YAML form for editing.

        Args:
            yaml_data (dict): The YAML data loaded from file.
            save_callback (function): Function to call after pressing 'Save'.
            main_menu_callback (function): Function to return to the main menu.
        """
        self.clear_frame()
        self.entries = {}

        tk.Label(self.root, text="Edit configuration", font=("Arial", 12, "bold")).pack(pady=5)
        container = tk.Frame(self.root)
        container.pack(fill="both", expand=True)

        canvas = tk.Canvas(container)
        canvas.pack(side="left", fill="both", expand=True)

        scrollbar = tk.Scrollbar(container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side="right", fill="y")

        canvas.configure(yscrollcommand=scrollbar.set)

        scroll_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
        scroll_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        row = 0
        for section, section_data in yaml_data.items():
            tk.Label(scroll_frame, text=section.upper(), font=("Arial", 10, "bold")).grid(row=row, column=0, sticky='w', pady=5)
            row += 1
            if isinstance(section_data, dict):
                for subkey, subval in section_data.items():
                    if isinstance(subval, dict):
                        tk.Label(scroll_frame, text=subkey + ":", font=("Arial", 9, "bold")).grid(row=row, column=0, sticky='w')
                        row += 1
                        for key, value in subval.items():
                            full_key = f"{section}.{subkey}.{key}"
                            self._render_entry(scroll_frame, full_key, key, value, row)
                            row += 1
                    else:
                        full_key = f"{section}.{subkey}"
                        self._render_entry(scroll_frame, full_key, subkey, subval, row)
                        row += 1

        if save_callback:
            tk.Button(self.root, text="Save and Execute", command=save_callback).pack(pady=10)

        if main_menu_callback:
            tk.Button(self.root, text="Back to Main Menu", command=main_menu_callback).pack()

    def _render_entry(self, parent, full_key, label, value, row):
        """Helper to render individual entry fields with optional tooltips and required field asterisk."""
        display_label = f"{label} *" if full_key in REQUIRED_FIELDS else label
        tk.Label(parent, text=display_label).grid(row=row, column=0, sticky='e')
        entry_widget = self._create_entry_widget(parent, label, value)
        entry_widget.grid(row=row, column=1, padx=5, pady=2)
        self.entries[full_key] = entry_widget

        desc = FIELD_DESCRIPTIONS.get(full_key, "No description available.")
        info_icon = tk.Label(parent, text="ℹ️", cursor="question_arrow")
        info_icon.grid(row=row, column=2, sticky='w')
        ToolTip(info_icon, desc)

    def _create_entry_widget(self, parent, key, value):
        """Create the correct entry widget for booleans, lists or plain strings."""
        if isinstance(value, bool):
            widget = ttk.Combobox(parent, values=["True", "False"], state="readonly", width=60)
            widget.set(str(value))
        else:
            widget = tk.Entry(parent, width=60)
            if isinstance(value, list) and key == "verbose":
                formatted = '[' + ', '.join([str(v).capitalize() for v in value]) + ']'
                widget.insert(0, formatted)
            else:
                widget.insert(0, str(value))
        return widget

    def render_path_selector(self, var, label_text, button_text, browse_callback):
        """Render an entry field with a file/folder selection button."""
        frame = tk.Frame(self.root)
        frame.pack(pady=10)

        tk.Label(frame, text=label_text).pack(side="left", padx=5)
        entry = tk.Entry(frame, textvariable=var, width=60)
        entry.pack(side="left", padx=5)

        browse_btn = tk.Button(frame, text=button_text, command=browse_callback)
        browse_btn.pack(side="left")

    def render_post_prediction_menu(self, stats, on_new_prediction, on_main_menu):
        """
        Render post-prediction options and display stats.

        Args:
            stats (list): A list of strings with statistical lines to show.
            on_new_prediction (function): Callback to predict new images.
            on_main_menu (function): Callback to return to main menu.
        """
        self.clear_frame()
        tk.Label(self.root, text="Prediction completed", font=("Arial", 14, "bold")).pack(pady=10)

        if stats:
            for line in stats:
                tk.Label(self.root, text=line, font=("Arial", 12)).pack()

        self.render_button("Predict new images", on_new_prediction)
        self.render_button("Back to Main Menu", on_main_menu)
        self.render_button("Exit", self.root.quit)

    def show_error(self, title, message):
        """Show a message box with an error."""
        messagebox.showerror(title, message)

    def browse_directory(self):
        """Open a folder selection dialog."""
        return filedialog.askdirectory(title="Select a folder")

    def show_wait_popup(self, message="Processing... Please wait."):
        """
        Displays a modal popup window to inform the user that a process is ongoing.

        Args:
            message (str): The message to display in the popup window. Defaults to "Processing... Please wait."
        """
        self.wait_popup = tk.Toplevel()
        self.wait_popup.title("Processing")
        self.wait_popup.geometry("300x100")
        self.wait_popup.resizable(False, False)

        label = tk.Label(self.wait_popup, text=message, padx=20, pady=20)
        label.pack()

        self.wait_popup.transient(self.root)
        self.wait_popup.grab_set()
        self.root.update()

    def close_wait_popup(self):
        """
        Closes the popup window if it exists and is currently displayed.
        """
        if hasattr(self, 'wait_popup') and self.wait_popup.winfo_exists():
            self.wait_popup.destroy()
            self.root.update()
