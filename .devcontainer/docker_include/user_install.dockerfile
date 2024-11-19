# Install external repos 
COPY .devcontainer/repos ${COLCON_WS}/src/repos

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Install PyTorch with specified versions and CUDA
RUN pip install --no-cache-dir \
    torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu118

# Install additional Python packages
RUN pip install --no-cache-dir \
    numpy \
    opencv-python \
    cv-bridge \
    scikit-image \
    matplotlib 

# RUN apt-get update && apt-get install -qq -y --no-install-recommends \
#     ros-humble-rmw-cyclonedds-cpp \
#     ros-humble-cyclonedds \
#     ros-humble-cv-bridge \
#     ros-humble-rviz2 \
#     && rm -rf /var/lib/apt/lists/*

# RUN pip install meson meson-python

# # Get the requirements file
# COPY requirements.txt ${COLCON_WS}/src/requirements.txt
# RUN pip install -r ${COLCON_WS}/src/requirements.txt && rm ${COLCON_WS}/src/requirements.txt

USER $USER

# Copy the script to checkout public git repos and make it executable
COPY .devcontainer/scripts/install_external_ros_packages.sh ${COLCON_WS}/src/install_external_ros_packages.sh
# Make the script executable and run it, then remove it
RUN /bin/bash -c '${COLCON_WS}/src/install_external_ros_packages.sh ${COLCON_WS}' && \
    sudo rm -f ${COLCON_WS}/src/install_external_ros_packages.sh && \
    sudo rm -f -r ${COLCON_WS}/src/repos

USER root