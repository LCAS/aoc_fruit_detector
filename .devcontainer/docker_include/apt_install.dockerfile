RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    git \
    ufw \
    wget \
    libcairo2-dev \
    pkg-config \
    python3-pip \
    python3-rosdep \
    python3-debian \
    libsystemd-dev \
    unattended-upgrades \
    usb-creator-gtk \
    libcups2-dev \
    screen-resolution-extra \
    cuda-toolkit-11-8 \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Remove unnecessary files or temporary files created during the setup
RUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
