# ============================================================
# File: main.py
# Author: Yael Vicente
# Date: 2025-May-06
# Description:
#   Main entry point for the YAML configuration GUI
#   for the AOC Fruit Detector project. Allows users to
#   configure training and prediction parameters using a
#   graphical interface, edit YAML files, and execute tasks.
# ============================================================

import os
import tkinter as tk
from gui_logic.form_renderer import FormRenderer
from gui_logic.yaml_handler import YAMLHandler
from gui_logic.prediction_manager import PredictionManager
from gui_logic.training_manager import TrainingManager
from config_paths import TRAINING_CONFIG_PATH, ROS_PARAMS_PATH, NON_ROS_PARAMS_PATH


class YAMLConfiguratorApp:
    """
    Main application class for the AOC Fruit Detector YAML Configurator GUI.
    Manages GUI flow, user interactions, and logic for loading, modifying,
    and saving YAML configuration files for training and prediction.
    """

    def __init__(self, root):
        """
        Initialize the GUI application and load the main menu.

        Args:
            root (tk.Tk): The root Tkinter window.
        """
        self.root = root
        self.root.title("AOC Fruit Detector - YAML Configurator")

        self.form_renderer = FormRenderer(self.root)
        self.yaml_handler = YAMLHandler()
        self.prediction_manager = PredictionManager(working_dir="/home/ros/fruit_detector_ws")
        self.training_manager = TrainingManager(working_dir="/home/ros/fruit_detector_ws")

        self.mode = None
        self.filepath = None
        self.yaml_data = None
        self.prediction_done = False
        self.transitioning_to_ros = False

        self.image_dir_var = tk.StringVar()
        self.pred_image_dir_var = tk.StringVar()
        self.pred_json_dir_var = tk.StringVar()

        self.show_main_menu()

    def show_main_menu(self):
        """Displays the main menu with mode selection buttons."""
        self.form_renderer.clear_frame()
        self.form_renderer.render_title("Select a mode:")
        self.form_renderer.render_button("Train new model", self.configure_training)
        self.form_renderer.render_button("Run predictions on images", self.select_prediction_mode)

    def configure_training(self):
        """Loads training configuration and renders the editable form."""
        self.mode = 'train'
        self.filepath = TRAINING_CONFIG_PATH
        self.yaml_data = self.yaml_handler.load_yaml(self.filepath)

        self.form_renderer.render_form(
            self.yaml_data,
            save_callback=self.save_and_train,
            main_menu_callback=self.show_main_menu
        )

    def select_prediction_mode(self):
        """Prompts the user to choose between ROS and non-ROS prediction modes."""
        self.form_renderer.clear_frame()
        self.form_renderer.render_title("Do you want to use ROS?")
        self.form_renderer.render_button("Yes, use ROS", lambda: self.configure_prediction(use_ros=True))
        self.form_renderer.render_button("No, without ROS", lambda: self.configure_prediction(use_ros=False))

    def configure_prediction(self, use_ros: bool, force_edit=False):
        """
        Configures the form for prediction parameters depending on ROS usage.

        Args:
            use_ros (bool): Whether ROS should be used.
            force_edit (bool): If True, allows editing config even after prediction.
        """
        self.transitioning_to_ros = use_ros
        self.mode = 'predict_ros' if use_ros else 'predict_non_ros'
        self.filepath = ROS_PARAMS_PATH if use_ros else NON_ROS_PARAMS_PATH

        self.yaml_data = self.yaml_handler.load_yaml(self.filepath)

        if not use_ros and self.prediction_done and not force_edit:
            self.ask_change_scope()
        else:
            self.form_renderer.render_form(
                self.yaml_data,
                save_callback=self.save_and_predict,
                main_menu_callback=self.show_main_menu
            )

    def ask_change_scope(self):
        """Offers the user the option to update either the entire config or only image paths."""
        self.form_renderer.clear_frame()
        self.form_renderer.render_title("What do you want to modify for the next prediction?")
        self.form_renderer.render_button("Edit full configuration file", lambda: self.configure_prediction(use_ros=False, force_edit=True))
        self.form_renderer.render_button("Only update image directory", self.show_image_dir_update_form)

    def show_image_dir_update_form(self):
        """
        Displays the interface for updating only the input and output directories used for prediction:
        - Test images directory
        - Predicted images output directory
        - Predicted JSON annotations directory
        """
        self.form_renderer.clear_frame()
        self.form_renderer.render_title("Update directories for prediction")

        self.form_renderer.render_path_selector(
            var=self.image_dir_var,
            label_text="Test image directory:",
            button_text="📂",
            browse_callback=lambda: self.browse_directory_to_var(self.image_dir_var)
        )

        self.form_renderer.render_path_selector(
            var=self.pred_image_dir_var,
            label_text="Predicted images output directory:",
            button_text="📂",
            browse_callback=lambda: self.browse_directory_to_var(self.pred_image_dir_var)
        )

        self.form_renderer.render_path_selector(
            var=self.pred_json_dir_var,
            label_text="Predicted JSON annotations directory:",
            button_text="📂",
            browse_callback=lambda: self.browse_directory_to_var(self.pred_json_dir_var)
        )

        self.form_renderer.render_button("Run predictions", self.save_and_predict_new_paths)
        self.form_renderer.render_button("Back to main menu", self.show_main_menu)

    def browse_directory_to_var(self, target_var):
        """
        Opens a file dialog to choose a directory and sets the result into the provided StringVar.

        Args:
            target_var (tk.StringVar): The variable to update with the selected path.
        """
        selected_dir = self.form_renderer.browse_directory()
        if selected_dir:
            target_var.set(selected_dir)

    def save_and_predict_new_paths(self):
        """
        Saves the new image directory and output paths, updates the config file, and runs the prediction.
        """
        image_dir = self.image_dir_var.get().strip()
        pred_image_dir = self.pred_image_dir_var.get().strip()
        pred_json_dir = self.pred_json_dir_var.get().strip()

        if not all(map(os.path.isdir, [image_dir, pred_image_dir, pred_json_dir])):
            self.form_renderer.show_error("Invalid path", "Please select valid directories for all fields.")
            return

        self.yaml_data['directories']['test_image_dir'] = image_dir
        self.yaml_data['directories']['prediction_output_dir'] = pred_image_dir
        self.yaml_data['directories']['prediction_json_dir'] = pred_json_dir
        self.yaml_handler.save_yaml(self.filepath, self.yaml_data)
        self.save_and_predict()

    def save_and_predict(self):
        """
        Saves the current configuration and executes the prediction script.
        """
        self.yaml_handler.update_yaml_from_entries(self.yaml_data, self.form_renderer.entries)
        self.yaml_handler.save_yaml(self.filepath, self.yaml_data)

        if self.mode == 'predict_ros':
            self.prediction_manager.run_ros_prediction()
            return

        self.form_renderer.show_wait_popup("Predicting images...\nPlease wait.")
        stats = self.prediction_manager.run_prediction(self.filepath)
        self.prediction_done = True
        self.form_renderer.close_wait_popup()

        self.form_renderer.render_post_prediction_menu(
            stats,
            on_new_prediction=self.select_prediction_mode,
            on_main_menu=self.show_main_menu
        )

    def save_and_train(self):
        """
        Saves the current training configuration and launches the training process.
        """
        self.yaml_handler.update_yaml_from_entries(self.yaml_data, self.form_renderer.entries)
        self.yaml_handler.save_yaml(TRAINING_CONFIG_PATH, self.yaml_data)
        self.training_manager.run_training()


if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("900x600")
    app = YAMLConfiguratorApp(root)
    root.mainloop()
