# ROSCon 2026

## ROS in the Field: Building Open Agri-Food Robotics Together

A hands-on session showcasing selected Agri-OpenCore components. Participants will be able to explore Dockerised examples and learn how reusable components can be incorporated into their own ROS-based systems.

### Contact
Workshop contact email: [Andy Perrett](mailto:APerrett@students.lincoln.ac.uk?subject=ROSCon26) (Make sure Subject line has ROSCon26 in the subject line). Clicking the name should open your mail client. 

### Documentation

[Documentation](scripts/documentation/README.md) is available for most scripts. You will also find improvements that need considering.

### Prerequisites 

Exploring the AOC Fruit detector repository requires [docker](https://docs.docker.com/get-docker/), the [VS Code IDE](https://code.visualstudio.com/Download), the dev container [extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers), and [Git](https://git-scm.com/install/) to be installed on your host laptop or PC. In the workshop, we will use a Linux laptop.

**A GPU is advised but not essential.** Running without a GPU will **severely slow everything down** though. Plus you will need to remove the `runtime: nvidia` from within docker-compose.yml file which is inside the .devcontainer. Do this immediately after the `git clone` below and before you open the container within VS Code.

Once the prerequisites are installed / satisifed:

1. **Clone the repository:** - We will use the `ROSCon` branch. Perhaps make a new folder:

    ```bash
    mkdir ROSCon
    cd ROSCon
    git clone --branch ROSCon --single-branch https://github.com/LCAS/aoc_fruit_detector.git
    ```

This creates an `aoc_fruit_detector` folder on the host. Open that folder in VS Code.

2. **Build and open the development container:**

    The Dev Containers extension may prompt you to **Reopen in Container** when it detects `.devcontainer/devcontainer.json`. If it does not, use the Command Palette (`Ctrl+Shift+P`) and run **Dev Containers: Reopen in Container**. VS Code then builds and starts the container. The repository is mounted inside it at `/home/ros/fruit_detector_ws/src`; the ROS workspace and build environment are inside the container.

    <span style="color: red;">**NOTE: This building process will take some time the first time you do this!**</span> There is a lot to download. Depending upon your internet connection, from a few minutes minutes to a lot more - coffee time.

    <span style="color: red;">Any errors I suggest asking chatGPT / other LLM. It is most likely your docker installation.</span>

3. **Access the desktop interface (optional):**

    *Optional, but without it you will not see any visualisations.*

    Open the **Ports** tab in VS Code and follow the forwarded desktop port shown there (the Dev Container configuration forwards port `5801`). Or just open a web browser an navigate to:
    ```
    https://localhost:5801
    ```
    Training graphs can be found at:
    ```
    https://localhost:6006
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

At this point it will be working if you can see data points in Rviz:

https://github.com/user-attachments/assets/b451f84f-ce7e-4e2a-a872-478c98d1e271

## Training a model

### Download

Training a model will involve downloading the [aoc_strawberry_dataset](https://lncn.ac/aocfruitds) zip file and extracting into the `scripts/data` folder.

### Config 

The `train_strawberry.py` script needs to be aware of the dataset:
```
    dataset = root / "data/strawberry"
    output = root / "data/training_output/strawberry_ripeness"
```

### Run

When the training script is run the training graphs can be viewed using `http://localhost:6006`.

The output files including the model weights will be saved in `scripts/data/training_output`.

```
python scripts/train_strawberry.py 
```

<img width="320" alt="loss_curve" src="https://github.com/user-attachments/assets/0a0774fb-dd7d-4b91-982d-3a268e79770f" />

### Detect

The `fruit_detection.py` that is called via the launch file (see above) is configured with the new model details just trained in `src/config/non_ros_params.yaml`:

```JSON
 model_file: '/home/ros/fruit_detector_ws/src/scripts/data/training_output/strawberry_ripeness/model_best.pth'
  test_metadata_catalog_file: '/home/ros/fruit_detector_ws/src/scripts/data/training_output/strawberry_ripeness/test_metadata_catalog.pkl'
  config_file: 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml'
```

Run as above by building, sourcing and then launching.