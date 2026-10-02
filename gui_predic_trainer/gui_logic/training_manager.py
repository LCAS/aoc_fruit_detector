# ============================================================
# File: training_manager.py
# Author: Yael Vicente
# Date: 2025-May-06
# Description:
#   Handles the execution of the training process for the
#   AOC Fruit Detector project. This includes workspace build,
#   environment sourcing, and training script execution via
#   external terminal.
# ============================================================

import subprocess

class TrainingManager:
    """
    Manages the training process by launching required commands
    inside a new terminal window.
    """

    def __init__(self, working_dir="/home/ros/fruit_detector_ws"):
        """
        Initialize the manager with the target workspace directory.

        Args:
            working_dir (str): Path to the catkin or colcon workspace.
        """
        self.working_dir = working_dir

    def run_training(self):
        """
        Launch a terminal to execute the training sequence.
        Executes:
            1. colcon build
            2. source install/setup.bash
            3. python3 train.py
        """
        try:
            subprocess.Popen([
                "xfce4-terminal",
                "-e",
                "bash -c 'cd {0} && colcon build && source install/setup.bash && python3 /home/ros/fruit_detector_ws/src/aoc_fruit_detector/scripts/predictor.py; exec bash'".format(self.working_dir)
            ])
        except Exception as e:
            raise RuntimeError(f"Failed to launch training process: {e}")