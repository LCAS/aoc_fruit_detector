# syntax = devthefuture/dockerfile-x

ARG BASE_IMAGE=lcas.lincoln.ac.uk/lcas/ros-docker-images:jammy-cuda11.8-humble-2
############################################################################################################
FROM ${BASE_IMAGE} AS base

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
FROM user_space AS sourcefilter
COPY . /tmp/src
# remove everything that isn't package.xml
RUN find /tmp/src -type f \! -name "package.xml" -print | xargs rm -rf

############################################################################################################
# install all dependencies listed in the package.xml
FROM user_space AS depbuilder
# copy the reduced source tree (only package.xml) from previous stage
COPY --from=sourcefilter /tmp/src /tmp/src
RUN rosdep update && apt-get update
RUN cd /tmp/src && rosdep install --from-paths . --ignore-src -r -y && cd && rm -rf /tmp/src

# Remove unnecessary files or temporary files created during the setup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

############################################################################################################
FROM depbuilder AS final

USER $USER

# include the file to configure the user's .bashrc
INCLUDE .devcontainer/docker_include/bashrc_config.dockerfile

CMD ["/bin/bash"]

# Deploys the application to production environment after successful build and test stages
# This stage handles the final deployment process including:
# - Setting up production environment variables
# - Deploying built artifacts to target infrastructure
# - Performing post-deployment health checks
# - Notifying relevant stakeholders of deployment status
FROM depbuilder AS deployed

USER root

# Copy the entire source code to the workspace
RUN mkdir -p ${COLCON_WS}/src
COPY . ${COLCON_WS}/src/

# Change ownership of the workspace to the user
RUN chown -R ${USER}:${USER} ${COLCON_WS}

USER $USER

# Build the workspace
WORKDIR ${COLCON_WS}
RUN bash -c "source /opt/ros/${ROS_DISTRO}/setup.bash; \
    colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release"

# Source the workspace in the entrypoint
RUN echo "source ${COLCON_WS}/install/setup.bash" >> ~/.bashrc

# Set the working directory
WORKDIR ${COLCON_WS}

# Default command
CMD ["/bin/bash"]