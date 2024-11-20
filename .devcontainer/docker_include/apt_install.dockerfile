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
    python3-dev \
    python3-opencv \
    libsystemd-dev \
    libgl1-mesa-glx \
    unattended-upgrades \
    usb-creator-gtk \
    libcups2-dev \
    screen-resolution-extra \
    cuda-toolkit-11-8 \
    ninja-build && \
    rm -rf /var/lib/apt/lists/*

# Set Python3 as default (if there is no link before)
#RUN ln -s /usr/bin/python3 /usr/bin/python
RUN if [ ! -e /usr/bin/python ]; then ln -s /usr/bin/python3 /usr/bin/python; fi

