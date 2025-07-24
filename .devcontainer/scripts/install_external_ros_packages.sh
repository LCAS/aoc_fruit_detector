#!/bin/bash

# Exit immediately if a command exits with a non-zero status and enable debug mode for verbose output
set -e -x

# Deactivate virtual environment to use system Python and avoid conflicts
# The base image has a virtual environment active in /opt/venv that needs to be deactivated
unset VIRTUAL_ENV
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
export PYTHONPATH=""

source /opt/ros/humble/setup.bash

# Update rosdep database to ensure all dependencies are up-to-date
rosdep update --rosdistro ${ROS_DISTRO}
sudo apt-get update 

# Navigate to the src directory of the current workspace
WS=$1
echo "WS dir ${WS}"
mkdir -p ${WS}/src/external_packages 
cd ${WS}/src/external_packages 

# Import repositories using vcs
echo "Importing repositories with vcs..."
vcs import < ${WS}/src/repos/external.repos 

# Install any dependencies specified in the cloned repositories' package.xml files
# The '-r' flag makes the process recursive
# The '-y' flag automatically answers "yes" to any prompts
rosdep install -r -y -i --from-paths .

# Install detectron2 in editable mode

#python3 -m pip install torch
sudo pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu118
sudo python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@754469e176b224d17460612bdaa2cb8112b04cd9'

echo "Setup complete."
