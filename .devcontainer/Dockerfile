# syntax = devthefuture/dockerfile-x

ARG BASE_IMAGE=lcas.lincoln.ac.uk/lcas/ros:jammy-humble-cuda11.8-opengl
############################################################################################################
FROM ${BASE_IMAGE} as base

USER root

ENV DEBIAN_FRONTEND=noninteractive
ENV ROS_DISTRO=humble
ARG user_id=1000
ENV user_id=${user_id}

ARG COLCON_WS=/home/ros/fruit_detector_ws
ENV COLCON_WS=${COLCON_WS}

# Install all required APT packages
INCLUDE .devcontainer/docker_include/apt_install.dockerfile

# Include env variables 
INCLUDE .devcontainer/docker_include/env_setup.dockerfile

############################################################################################################
FROM base AS user_space

INCLUDE .devcontainer/docker_include/user_install.dockerfile

############################################################################################################
# get the source tree and analyse it for its package.xml only
FROM user_space as sourcefilter
COPY . /tmp/src
# remove everything that isn't package.xml
RUN find /tmp/src -type f \! -name "package.xml" -print | xargs rm -rf

############################################################################################################
# install all dependencies listed in the package.xml
FROM user_space as depbuilder
# copy the reduced source tree (only package.xml) from previous stage
COPY --from=sourcefilter /tmp/src /tmp/src
RUN rosdep update && apt-get update
RUN cd /tmp/src && rosdep install --from-paths . --ignore-src -r -y && cd && rm -rf /tmp/src

# Remove unnecessary files or temporary files created during the setup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

############################################################################################################
FROM depbuilder as final

USER $USER

# include the file to configure the user's .bashrc
INCLUDE .devcontainer/docker_include/bashrc_config.dockerfile

CMD ["/bin/bash"]
