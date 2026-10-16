# Fruit Detection (`fruit_detection.py`)

## Overview

`fruit_detection.py` implements the ROS 2 node `aoc_fruit_detector`, using `DetectronPredictor` to detect strawberries or tomatoes in RGB or RGB-D images. It supports live ROS 2 camera topics and offline processing of image files. In ROS mode, detections are converted into `FruitInfoArray` messages with segmentation, bounding boxes, confidence, 2D orientation and estimated 3D position. Optional outputs include annotated images and RViz markers.

## Dependencies

- ROS 2 Python: `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `visualization_msgs`, `tf2_ros`, `ament_index_python`
- ROS interfaces: `aoc_fruit_detector.msg.FruitInfoMessage`, `FruitInfoArray`
- Vision: `opencv-python` (`cv2`), `cv_bridge`, `image_geometry`, `numpy`
- Predictor: `detectron_predictor.detectron_predictor.DetectronPredictor` and `FruitTypes`
- Other imports: `PyYAML`, `scipy`, `matplotlib`

The script requires the custom detector package, message definitions and a YAML configuration file. Model architecture, training, and inference thresholds are controlled by the external predictor/configuration, not defined here.

## Running

```bash
python3 fruit_detection.py --config-file /path/to/config.yaml --ros-args -p use_ros:=true
```

For offline images, set `use_ros:=false`. The `--config-file` argument is mandatory in both modes. The script parses ROS arguments separately using `parse_known_args()`.

## YAML configuration

The following keys are read directly by this script:

```yaml
directories:
  test_image_dir: /path/to/images
  prediction_json_dir: /path/to/json
  prediction_output_dir: /path/to/output
settings:
  fruit_type: strawberry  # or tomato
  filename_patterns:
    rgb: RGB
    depth: DEPTH
```

The pattern strings are illustrative, not prescribed filenames. The predictor may require additional configuration keys. Under `files` and `directories`, paths starting with `./` are resolved relative to the installed `aoc_fruit_detector` package share directory.

## ROS parameters

| Parameter | Default | Purpose |
|---|---|---|
| `use_ros` | `true` | Select live ROS or offline image processing |
| `min_depth` | `0.1` | Minimum accepted live depth, in metres, for 3D pose estimation |
| `max_depth` | `15.0` | Exclusive maximum accepted live depth, in metres, for 3D pose estimation |
| `constant_depth_value` | `1.0` | Substitute depth when no live depth image has arrived; still subject to the depth limits |
| `fruit_type` | `strawberry` | Selects tomato/strawberry metadata in ROS mode |
| `pose3d_frame` | `''` | Frame label assigned to the estimated 3D pose; camera info does not change it |
| `pose3d_tf` | `false` | Apply optical-to-camera transform to computed 3D point |
| `verbose` | `[false, false, false, true, true]` | Draw centroid, box, mask, coordinate axes, angle text |
| `pub_verbose` | `false` | Publish composed image and include image messages in fruit array |
| `pub_markers` | `false` | Publish 3D visualization markers |

**Important:** YAML `settings.fruit_type` determines the predictor's `FruitTypes` value, while the ROS `fruit_type` parameter independently determines biological metadata. Configure both consistently.

## ROS interfaces

| Direction | Topic | Message type | Description |
|---|---|---|---|
| Subscribe | `camera/image_raw` | `sensor_msgs/Image` | RGB input (converted to BGR8) |
| Subscribe | `camera/depth` | `sensor_msgs/Image` | Depth input (converted to 32FC1) |
| Subscribe | `camera/camera_info` | `sensor_msgs/CameraInfo` | Camera calibration and intrinsics |
| Publish | `fruit_info` | `aoc_fruit_detector/FruitInfoArray` | Fruit detections and attributes |
| Publish | `image_composed` | `sensor_msgs/Image` | Optional annotated RGB image |
| Publish | `fruit_markers` | `visualization_msgs/MarkerArray` | Optional ripeness-colored sphere markers |

Subscriptions use `qos_profile_sensor_data`; publishers use queue depth 5. Topic names are relative to the node namespace.

## Processing pipeline

### Live ROS mode

1. `depth_callback()` accepts `16UC1` (converted from millimetres to metres) or `32FC1` depth images, and marks non-finite or non-positive values invalid.
2. `camera_info_callback()` updates the pinhole camera model only. Until a camera-info message arrives, hard-coded calibration is used: 1280×720, `fx=fy=699.05`, `cx=639.74`, `cy=374.974`. The output frame remains the value of `pose3d_frame`.
3. `image_callback()` looks up the transform from `zed_left_camera_optical_frame` to `zed_camera_link`, falling back to a fixed rotation if unavailable.
4. Convert the incoming RGB image to BGR8; use the latest stored depth image, resizing it to the RGB dimensions if necessary, or fill with `constant_depth_value`.
5. Stack BGR and depth into a four-channel array, and invoke `DetectronPredictor.get_predictions_message()`.
6. Extract annotations, confidence values, orientation/centroids and category names from the predictor result.
7. Estimate each detection's 3D pose from valid depth in a 5×5 patch around its centroid. Detections with out-of-bounds centroids or fewer than three valid depth samples are skipped.
8. Construct a `FruitInfoMessage` for each remaining annotation and publish the resulting `FruitInfoArray`.
9. Optionally draw overlays and publish a `MarkerArray`.

**Timing:** RGB and depth are not time-synchronised; the most recently received depth frame is used. Image IDs are formed by concatenating the RGB message's seconds and nine-digit nanoseconds.

### Offline mode

1. List files from `directories.test_image_dir`.
2. Select RGB and depth filenames containing the configured patterns.
3. Match a depth filename by replacing the first RGB pattern occurrence in the RGB filename.
4. With matching depth, load BGR and unchanged depth, replace invalid depth values, stack into RGB-D and call `get_predictions_image()`.
5. Without matching depth, warn and call `get_rgb_predictions_image()`.
6. Write prediction JSON/output files through the external predictor.

The offline path runs inside the constructor; `main()` subsequently calls `rclpy.spin()` even when `use_ros` is false.

## Detection message contents

Each `FruitInfoMessage` receives:

- `fruit_id`, `image_id`, `header`, `confidence` and detected `ripeness_category`.
- Biological descriptors for strawberry or tomato (fixed strings, not predicted properties).
- `bbox`, polygon `mask2d`, annotation `area`, centroid and orientation (`pose2d`).
- A depth-derived `pose3d` with identity quaternion.
- Placeholder values for quality, ripeness level, volume, bounding volume, 3D mask and occlusion.

For a category named `fruit_ripe`, ripeness level is set to `0.95`; otherwise the script assigns `0.15` to the private `_ripeness_level` attribute rather than `ripeness_level`.

## 3D pose calculation

`compute_pose2d()` uses the annotation's centroid and orientation; missing entries produce NaNs. `compute_pose3d()` takes the median of finite, positive samples in a 5×5 depth patch around the integer centroid, accepting only depths from `min_depth` (inclusive) to `max_depth` (exclusive). At least three valid samples are required; otherwise that detection is omitted from the published fruit array. The median depth is treated as optical-axis Z: the pixel ray from `PinholeCameraModel.projectPixelTo3dRay()` is scaled by `depth / ray[2]`. With `pose3d_tf=true`, the homogeneous point is transformed using the lookup (or fallback) optical-to-camera transform. The resulting pose uses an identity quaternion and the `pose3d_frame` parameter as its frame label; camera info does not set that label.

## Visualization

`add_markers_on_image()` can overlay centroid dots, polygon outlines, bounding boxes, oriented coordinate axes and angle labels. Green denotes ripeness level below 0.5, red denotes 0.5 or above. `create_fruit_marker()` generates spherical RViz markers with 0.02 m diameter and the same green/unripe, red/ripe colors. Each published marker array starts with `Marker.DELETEALL`, clearing markers left over from earlier detections; the active markers have unlimited lifetime.

## Implementation caveats

- **No synchronization:** stale depth may be paired with a newer RGB frame; resizing does not ensure spatial alignment.
- **Potential uninitialized variable:** offline `prediction_json_output_file` is only assigned when `prediction_json_dir` is nonempty, but is passed to the predictor unconditionally.
- **Potential uninitialized variable:** `self.tomato` is only assigned for exactly `tomato` or `strawberry` in the ROS parameter; other values may fail later.
- **Overlay dependency:** `add_text=true` uses `end_x_x` and `end_y_x`, which are only defined if `draw_cf=true`.
- **Placeholder metadata:** `volume=area*2`, `bvol=bbox`, `mask3d=segmentation`, fixed quality/occlusion and hard-coded botanical metadata are not measured quantities.
- **3D frame consistency:** `pose3d_frame` is only a label: coordinates remain in the optical frame when `pose3d_tf=false`, and are transformed to the hard-coded `zed_camera_link` frame when it is true. Set the label to match the selected coordinate frame. The TF lookup uses hard-coded ZED frame names.
- **ROS and offline configurations differ:** the YAML fruit type controls the predictor, but the ROS parameter controls published biological labels.

## Main functions

| Function | Responsibility |
|---|---|
| `__init__()` | Parameters, YAML, predictor, ROS interfaces or offline batch |
| `image_callback()` | Main live detection and publication pipeline |
| `depth_callback()` | Decode supported depth encodings and mark invalid samples |
| `camera_info_callback()` | Update camera calibration |
| `compute_pose2d()` | Annotation centroid/orientation to `Pose2D` |
| `compute_pose3d()` | Depth and pixel ray to `PoseStamped` |
| `get_optic_tf()` | Resolve optical-to-camera transform |
| `add_markers_on_image()` | Annotate RGB image |
| `publish_fruit_markers()` | Publish 3D marker array |
| `main()` | Parse arguments, initialize and spin ROS 2 node |

---
