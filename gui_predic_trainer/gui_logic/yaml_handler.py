# ============================================================
# File: yaml_handler.py
# Author: Yael Vicente
# Date: 2025-May-06
# Description:
#   Provides functions to load and save YAML configuration files
#   using ruamel.yaml with support for preserving formatting and
#   comments. Designed for integration with the AOC Fruit Detector GUI.
# ============================================================

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedSeq
from tkinter import messagebox


class YAMLHandler:
    """
    Class for loading and saving YAML files with preserved structure and formatting.
    """

    def __init__(self):
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

    def load_yaml(self, filepath: str) -> dict:
        """
        Load a YAML file from disk.

        Args:
            filepath (str): Path to the YAML file.

        Returns:
            dict: Parsed YAML data.

        Raises:
            Exception: If loading fails.
        """
        try:
            with open(filepath, 'r') as file:
                data = self.yaml.load(file)
            return data
        except Exception as e:
            raise Exception(f"Failed to load YAML file: {e}")

    def save_yaml(self, filepath: str, data: dict):
        """
        Save YAML data back to disk.

        Args:
            filepath (str): Path where to save the YAML file.
            data (dict): Data to write.

        Raises:
            Exception: If writing fails.
        """
        try:
            with open(filepath, 'w') as file:
                self.yaml.dump(data, file)
        except Exception as e:
            raise Exception(f"Failed to save YAML file: {e}")

    def update_yaml_from_entries(self, yaml_data: dict, entries: dict):
        """
        Updates the yaml_data dictionary based on the current state of the GUI entry widgets.

        Args:
            yaml_data (dict): Original YAML dictionary to update.
            entries (dict): Dictionary of tkinter Entry or Combobox widgets keyed by YAML path.
        """
        for key, widget in list(entries.items()):
            try:
                if not widget.winfo_exists():
                    continue
            except:
                continue

            parts = key.split('.')
            value = widget.get()

            if parts[-1] == "verbose":
                try:
                    value = [v.strip().capitalize() == 'True' for v in value.strip('[]').split(',')]
                    value = CommentedSeq(value)
                    value.fa.set_flow_style()
                except Exception:
                    messagebox.showerror("Error", "Invalid value for 'verbose'. Expected a list like [True, False, ...]")
                    return
            elif value.lower() in ['true', 'false']:
                value = value.lower() == 'true'
            else:
                try:
                    value = int(value)
                except ValueError:
                    try:
                        value = float(value)
                    except ValueError:
                        pass

            ref = yaml_data
            try:
                for part in parts[:-1]:
                    ref = ref[part]
                ref[parts[-1]] = value
            except Exception as e:
                messagebox.showerror("Error", f"Failed to update value for key '{key}': {e}")
