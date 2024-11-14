# Install external repos 
COPY .devcontainer/repos ${COLCON_WS}/src/repos

RUN apt-get update && \
    apt-get install -y libcairo2 libcairo2-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

RUN pip install meson meson-python

# Get the requirements file
COPY requirements.txt ${COLCON_WS}/src/requirements.txt
RUN pip install -r ${COLCON_WS}/src/requirements.txt

USER $USER

# Copy the script to checkout public git repos and make it executable
COPY .devcontainer/scripts/install_external_ros_packages.sh ${COLCON_WS}/src/install_external_ros_packages.sh
# Make the script executable and run it, then remove it
RUN /bin/bash -c '${COLCON_WS}/src/install_external_ros_packages.sh ${COLCON_WS}' && \
    sudo rm -f ${COLCON_WS}/src/install_external_ros_packages.sh && \
    sudo rm -f -r ${COLCON_WS}/src/repos

USER root