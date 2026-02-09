# Dockerfile for nuclio function deployment
FROM ros:humble-base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV ROS_DISTRO=humble
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-opencv \
    python3-numpy \
    python3-pil \
    python3-colcon-common-extensions \
    ros-humble-cv-bridge \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-visualization-msgs \
    && rm -rf /var/lib/apt/lists/*

# Create workspace for AOC messages
WORKDIR /opt/ros_ws
RUN mkdir -p src/aoc_fruit_detector

# Copy package files for building messages
COPY ../CMakeLists.txt src/aoc_fruit_detector/
COPY ../package.xml src/aoc_fruit_detector/
COPY ../msg src/aoc_fruit_detector/msg/

# Build the AOC message package
RUN bash -c "source /opt/ros/humble/setup.bash && colcon build --packages-select aoc_fruit_detector"

# Set working directory for function
WORKDIR /opt/nuclio

# Copy function files
COPY main.py .
COPY ros2_bridge.py .
COPY cvat_converter.py .
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Set up environment for ROS2
ENV ROS_DOMAIN_ID=0
ENV RMW_IMPLEMENTATION=rmw_cyclonedx_cpp
ENV PYTHONPATH="/opt/ros_ws/install/aoc_fruit_detector/lib/python3.10/site-packages:${PYTHONPATH}"

# Source both ROS2 and AOC messages setup
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /opt/ros_ws/install/setup.bash" >> ~/.bashrc

# Expose port for nuclio
EXPOSE 8080

# Set entrypoint
CMD ["/bin/bash", "-c", "source /opt/ros/humble/setup.bash && source /opt/ros_ws/install/setup.bash && python3 main.py"]