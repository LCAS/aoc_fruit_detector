#!/bin/bash

set -e -x
sudo apt-get update
sudo apt-get install net-tools -y 
sudo ifconfig lo multicast || true

sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev lo || true


# Creates a colcon configuration directory and sets up default build options
# - Creates ~/.colcon directory if it doesn't exist
# - Generates a defaults.yaml file with symlink-install enabled for faster builds
# - Symlink installation avoids copying files, making iterative development more efficient
mkdir -p ${HOME}/.colcon
cat << EOF > ${HOME}/.colcon/defaults.yaml
{
    "build": {
        "symlink-install": true
    }
}
EOF

# network hack for DDS

rosdep update --rosdistro ${ROS_DISTRO}
rosdep install -y --from-paths src/ -i --rosdistro ${ROS_DISTRO} -r

colcon build --continue-on-error
