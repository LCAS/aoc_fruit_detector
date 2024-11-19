#!/bin/bash

# Exit immediately if a command exits with a non-zero status and enable debug mode for verbose output
set -e -x

# Update rosdep database to ensure all dependencies are up-to-date
rosdep update
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

# Install Python dependencies for the ultralytics_ros package
python3 -m pip install -e detectron2

echo "Setup complete."
