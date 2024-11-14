# This Dockerfile sets up a ROS 2 development environment. It starts by setting the environment variable USER to 'ros'.
# The 'chpasswd' command sets the password for the 'ros' user to 'ros'. The 'adduser' command adds the 'ros' user to the 'sudo' group.
# A directory for the ROS 2 workspace is created at the path specified by the COLCON_WS environment variable.
# The working directory is set to /home/ros, and ownership and permissions for this directory are adjusted to ensure the 'ros' user has the necessary access.
ENV USER=ros
RUN echo 'ros:ros' | chpasswd
RUN adduser ${USER} sudo
RUN mkdir -p ${COLCON_WS}/src/
WORKDIR /home/ros
RUN chown -R ros:ros /home/ros
RUN chmod 755 /home/ros

# ENVs
ENV HOME=/home/ros
ENV PATH="/home/ros/.local/bin:${PATH}"
# Force Python stdout and stderr streams to be unbuffered.
ENV PYTHONUNBUFFERED=1
# Set the LD_PRELOAD environment variable
ENV LD_PRELOAD=/usr/lib/libdlfaker.so:/usr/lib/libvglfaker.so
# enable all capabilities for the container
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Add the current user to the dialout, video, and tty groups.
# This grants the user permissions to access serial ports, video devices, and terminal devices respectively.
RUN usermod -a -G dialout $USER && \
    usermod -a -G video $USER && \
    usermod -a -G tty $USER