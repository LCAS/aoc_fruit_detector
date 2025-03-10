# AOC Fruit Detector using Detectron2 MaskRCNN

Instance segmentation of a scene and output Mask-RCNN predictions as images and json message/file (Agri-OpenCore), fully ROS2 integrated system.

![Example images](./scripts/data/figure/output_fig.png)

## Installation, Requirements and Running

### Without Docker Installation

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

### Running

Run following to publish/save annotations detected by `aoc_fruit_detector` package.

```bash
ros2 launch aoc_fruit_detector fruit_detection.launch.py
```

## Parameters

The **config** folder contains two parameter files for specifying system characteristics: 
* **ros_params.yaml** allows tuning of ROS framework parameters.
* **non_ros_params.yaml** contains parameters for the fruit detection module.

### Key ROS2 parameters

* **min_depth, max_depth**: Define the minimum and maximum depth values for the depth channel of the camera input.
* **constant_depth_value**: Used when no depth image or channel is available. This value is assumed as the distance between the camera and the detected fruit, enabling 3D pose estimation. This is particularly useful for RGB cameras without depth estimation capabilities. The default value is 1.0 m.
* **fruit_type**: Specifies the type of fruit to detect. Currently supported values are **"strawberry"** and **"tomato"**.
* **pose3d_frame**: Sets the frame ID for the 3D poses of the detected fruits.
* **pose3d_tf**: In some cases, camera_info message includes camera_frame rather than camera_optical_frame. In this case, the 3D pose estimated should be transformed from optical frame to camera frame. For transformation-needed cases, set it to `True`, otherwise keep it as `False`. 
* **pub_verbose**: Publishes an annotated image as a `sensor_msgs/Image` message in the ROS2 framework.
* **verbose**: Determines which annotations appear on the annotated image. A Boolean list specifies the visualization of the following annotations in order: `[centroid, bounding box, mask, coordinate frames, text]`.
* **pub_markers**: To publish RViz markers in the ROS2 framework.
* **use_ros**: Specifies whether the fruit detection framework is integrated with the ROS2 framework.
  * If `False`, images are taken from the `test_image_dir` directory specified in **non_ros_params.yaml**, and the annotated images and JSON annotation files are saved to `prediction_output_dir` and `prediction_json_dir` directories, respectively.
  * If `True`, RGB and depth images are subscribed from camera topics (`/camera/image_raw`, `/camera/depth` and `/camera/camera_info`), and annotations are published via ROS2 topics (`/fruit_info` and `/image_composed`) and as RViz markers (`/fruit_markers`).
      * These topics can be remapped in `fruit_detection.launch.py` file.
      * Input topics:
          * `/camera/image_raw`: Maps to the rectified colour image topic provided by the camera. Using rectified colour images is important for 3D pose estimation accuracy.
          * `/camera/depth`: Maps to the depth image topic of the camera. Using registered depth images is important for 3D pose estimation accuracy.
          * `/camera/camera_info`: Maps to the camera intrinsic calibration information. For 3D pose estimation, camera intrinsic calibration information is essential. A default pinhole camera model is defined in the package, but for accurate pose estimation provide camera calibration data to the system.
      * Output topics:
          * `/fruit_info`: Contains information about detected fruits, including their positions and classifications. The message type is **FruitInfoArray**, a specific message type for the `aoc_fruit_detector` package.
          * `/image_composed`: Publishes the composed image with annotations overlaid, such as bounding boxes and labels for detected fruits. The message type is **sensor_msgs/Image**. This message is not published if the `pub_verbose` parameter is False.
          * `/fruit_markers`: Publishes 3D markers for detected fruits to be visualized in RViz. The message type is **visualization_msgs/MarkerArray**.

### Key AOC Fruit Detector parameters

* **download_assets**: If assets such as model and datasets should be downloaded, set it to True. Running system properly in the first trial, it is recommended to download assets.
* **rename_pred_images**: To rename the predicted images in img_000001.png like format, set this parameter to True.
* **segm_masks**: To visualise segmentation mask on the annotated images, set this parameter to True.
* **bbox**: To visualise bounding box mask on the annotated images, set this parameter to True.
* **show_orientation**: To visualise orientation information on the annotated images, set this parameter to True.
* **fruit_type**: Specifies the type of fruit to detect. Currently supported values are **"strawberry"** and **"tomato"**.
* **confidence_threshold**: To remove annotations with low confidence score, define a threshold within the valid range [0, 1] by setting this parameter. 
* **filename_patterns**: To set the filename pattern for RGB and depth images use **rgb** and **depth** clusters of the parameter. 

Please note that annotated images and JSON annotation files are saved to the location where the package is installed. Check the **install** folder of the repository to find the annotation outputs.

## Wiki

To get more information about the AOC Fruit Detector, please refer to [Wiki page][wiki_page]

---

[wiki_page]: https://github.com/LCAS/aoc_fruit_detector/wiki
