# AOC Fruit Detector using Detectron2 MaskRCNN

Instance segmentation of a scene and output Mask-RCNN predictions as images and json message/file (Agri-OpenCore), fully ROS2 integrated system.

![Example images](./scripts/data/figure/output_fig.png)

## Installation, Requirements and Running

The docker container automatically installs all required dependencies. The main dependencises are listed below. However, without dockerisation, the following required packages or check/install required versions should be installed from requirements.txt file.

`python3` `torchvision` `pickle` `numpy` `opencv-python` `cv-bridge` `scikit-image` `matplotlib`
`detectron2` 

```
pip install -r requirements.txt
```
Detectron2 package is also automatically installed by Docker container. However, without dockerisation, it can be cloned from GitHub and installed into the workspace.

```
git clone https://github.com/facebookresearch/detectron2.git
python3 -m pip install -e detectron2
```

### Dockerised Installation

1. **Open in Visual Studio Code:**

    Open the cloned repository in VSCode. VSCode will prompt you to "Reopen in Container." Alternatively, you can use the command palette (`Ctrl+Shift+P`) and search for the "reopen in container" command.

   <img src="https://github.com/LCAS/ros2_pkg_template/assets/47870260/52b26ae9-ffe9-4e7c-afb9-88cee88f870f" width="300">

2. **Accessing the Desktop Interface:**

    Open the user interface by navigating to the PORTS tab in VSCode, selecting port `6080` (or port `5801` for the CUDA-OpenGL version), and opening it in the browser.

   <img src="https://github.com/LCAS/ros2_pkg_template/assets/47870260/b61f4c95-453b-4c92-ad66-5133c91abb05" width="400">

3. **Getting Started: Build Packages**

    Build package with

    ```bash
    cd ${your_ws} && colcon build
    source install/setup.bash 
    ```

### Dockerised Running

Run following to publish annotations detected by `aoc_fruit_detector` package.

```bash
ros2 launch aoc_fruit_detector fruit_detection.launch.py
```

If there is no depth channel, you may use a dummy depth value, default value is 1.0 m. Depth channel is used to estimate 3D pose of the fruits.

```bash
ros2 launch aoc_fruit_detector fruit_detection.launch.py constant_depth_value:=0.5
```

## Wiki

To get more information about the AOC Fruit Detector, please refer to [Wiki page][wiki_page]

---

[wiki_page]: https://github.com/LCAS/aoc_fruit_detector/wiki