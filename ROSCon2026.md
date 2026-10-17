# ROSCon 2026

## ROS in the Field: Building Open Agri-Food Robotics Together

A hands-on session showcasing selected Agri-OpenCore components. Participants will be able to explore Dockerised examples and learn how reusable components can be incorporated into their own ROS-based systems.



### Prerequisites 

Exploring the AOC Fruit detector repository requires [docker](https://docs.docker.com/get-docker/), the [VS Code IDE](https://code.visualstudio.com/Download), and [Git](https://git-scm.com/install/) to be installed on your host laptop or PC. In the workshop, we will use a Linux laptop.

Once they are installed:

1. **Clone the repository:** - We will use the ROSCon branch. Perhaps make a new folder 
    ```bash
    mkdir ROSCon
    cd ROSCon
    git clone --branch ROSCon --single-branch https://github.com/LCAS/aoc_fruit_detector.git
    ```

This creates an `aoc_fruit_detector` folder on the host. Open that folder in VS Code.

2. **Build and open the development container:**

    The Dev Containers extension may prompt you to **Reopen in Container** when it detects `.devcontainer/devcontainer.json`. If it does not, use the Command Palette (`Ctrl+Shift+P`) and run **Dev Containers: Reopen in Container**. VS Code then builds and starts the container. The repository is mounted inside it at `/home/ros/fruit_detector_ws/src`; the ROS workspace and build environment are inside the container.

3. **Access the desktop interface (optional):**

    *Optional, but without it you will not see any visualisations.*

    Open the **Ports** tab in VS Code and follow the forwarded desktop port shown there (the Dev Container configuration forwards port `5801`). Or just open a web browser an navigate to:
    ```
    https://localhost:5801
    ```

### Running

Run the following to view published topics/saved annotations detected by `aoc_fruit_detector` package. This will also start Rviz in the web browser desktop view:

You want to be inside the container (in VS Code terminal or web browser desktop view terminal):

```bash
cd ~/fruit_detector_ws/src
```

For first run *and* after making any changes to any file you need to build and source from the above folder:
```bash
colcon build
source install/setup.bash
```

Then to run:
```bash
ros2 launch aoc_fruit_detector fruit_detection_with_bag.launch.py
```

## What you should see

At this point it will be working if you can see: