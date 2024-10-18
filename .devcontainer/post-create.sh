#!/bin/bash


function add_config_if_not_exist {
    if ! grep -F -q "$1" $HOME/.bashrc; then
        echo "$1" >> $HOME/.bashrc
    fi
}

add_config_if_not_exist "export ROS_DOMAIN_ID=9"

add_config_if_not_exist "source /opt/ros/humble/setup.bash"
LOCAL_SETUP_FILE=`pwd`/install/setup.bash
add_config_if_not_exist "if [ -r $LOCAL_SETUP_FILE ]; then source $LOCAL_SETUP_FILE; fi"

source $HOME/.bashrc

source /opt/ros/humble/setup.bash
colcon build --symlink-install --continue-on-error
