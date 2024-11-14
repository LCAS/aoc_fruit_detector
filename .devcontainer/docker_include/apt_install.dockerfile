RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    git \
    wget \
    python3-pip \
    python3-rosdep \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cyclonedds \
    ninja-build

# Install required system libraries for systemd
RUN apt-get update && \
    apt-get install -y libsystemd-dev && \
    rm -rf /var/lib/apt/lists/*

# Remove unnecessary files or temporary files created during the setup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
