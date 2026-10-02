# ============================================================
# File: prediction_manager.py
# Author: Yael Vicente
# Date: 2025-May-06
# Description:
#   Handles prediction execution using subprocess calls.
#   Parses prediction logs and returns summary statistics.
#   Designed to integrate with the AOC Fruit Detector GUI.
# ============================================================

import subprocess
import re
import os


class PredictionManager:
    """
    Manages the execution of the prediction script and parses output statistics.
    """

    def __init__(self, working_dir="/home/ros/fruit_detector_ws"):
        """
        Initialize the prediction manager.

        Args:
            working_dir (str): Working directory where the prediction script resides.
        """
        self.working_dir = working_dir

    def run_prediction(self, config_file_path, use_ros=False):
        """
        Executes the prediction script with the given configuration file.

        Args:
            config_file_path (str): Path to the YAML configuration file.
            use_ros (bool): Whether to run with ROS parameters.

        Returns:
            list[str]: Lines of log output containing prediction statistics.

        Raises:
            RuntimeError: If the prediction process fails.
        """
        # Build the command as a single string
        command = (
            "colcon build && "
            "source install/setup.bash && "
            f"python3 src/aoc_fruit_detector/scripts/fruit_detection.py "
            f"--config-file {config_file_path} "
        )
        if not use_ros:
            command += "--ros-args -p use_ros:=false"

        try:
            result = subprocess.run(
                ["bash", "-c", command],
                cwd=self.working_dir,
                capture_output=True,
                text=True,
                check=True,
            )
            print(result.stdout)
            print(result.stderr)
            return self._parse_statistics(result.stdout, config_file_path=config_file_path)

        except subprocess.CalledProcessError as e:
            print("[ERROR] Prediction process failed:")
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)

            from tkinter import messagebox
            messagebox.showerror("Error", "The prediction failed. Please verify that the image path is correct..")
            raise RuntimeError("Predictions failed")


    def _parse_statistics(self, log_output, config_file_path=None):
        """
        Parses prediction statistics from the log output and appends output directories.

        Args:
            log_output (str): The full stdout from the prediction subprocess.
            config_file_path (str, optional): Path to the YAML configuration file to extract output paths.

        Returns:
            list[str]: List of formatted statistic lines including output paths.
        """
        stats = []
        for line in log_output.splitlines():
            if re.match(r"Total processed images:\s+\d+", line):
                stats.append(line.strip())
            elif re.match(r"Images with predictions:\s+\d+", line):
                stats.append(line.strip())
            elif re.match(r"Images without predictions:\s+\d+", line):
                stats.append(line.strip())

        if config_file_path and os.path.exists(config_file_path):
            try:
                import yaml
                with open(config_file_path, 'r') as f:
                    config = yaml.safe_load(f)
                    json_dir = config.get('directories', {}).get('prediction_json_dir', '[Not defined]')
                    image_dir = config.get('directories', {}).get('prediction_output_dir', '[Not defined]')

                    abs_pred_dir = os.path.abspath(image_dir)
                    abs_json_dir = os.path.abspath(json_dir)

                    stats.append(f"Predicted JSON files saved to: {abs_json_dir}")
                    stats.append(f"Predicted images saved to: {abs_pred_dir}")
            except Exception as e:
                stats.append(f"[Warning] Failed to read output paths from YAML: {e}")

        return stats
    
    def run_ros_prediction(self):
        """
        Executes the ROS 2 launch command to perform fruit detection using ROS.

        This method:
        - Builds the workspace using colcon.
        - Sources the ROS 2 environment.
        - Launches the Detectron2 fruit detection node via ROS 2.

        The process is executed in a new xfce4-terminal window.
        """
        launch_command = (
            "cd /home/ros/fruit_detector_ws && "
            "colcon build && "
            "source install/setup.bash && "
            "ros2 launch aoc_fruit_detector fruit_detection.launch.py; exec bash"
        )

        try:
            subprocess.Popen([
                "xfce4-terminal",
                "--title=ROS2 Prediction",
                "-e", f"bash -c '{launch_command}'"
            ])
        except Exception as e:
            raise RuntimeError(f"Failed to launch ROS 2 prediction: {e}")

