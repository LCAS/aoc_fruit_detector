# Fruit Detection — `predictor.py`

## Purpose

`predictor.py` is a **non-ROS example integration script** for Detectron-based fruit detection and segmentation. It reads test RGB images, pairs them with depth images where available, constructs four-channel RGBD inputs, requests predictions, and yields annotation data. After processing the images, it invokes the model evaluation workflow.



## Dependencies

- Python packages: `opencv-python` (`cv2`), `numpy`, `PyYAML` (`yaml`).
- Project modules:
  - `detectron_predictor.detectron_predictor.DetectronPredictor`
  - `detectron_trainer.detectron_trainer.DetectronTrainer`
  - `detectron_predictor.json_writer.pycococreator.pycococreatortools.fruit_orientation.FruitTypes`
  - `utils.utils.LearnerUtils`, `find_path`, and `find_workspace_root`
- Standard library: `os`, `logging`, and `traceback`.

These project modules and their underlying model configurations must be available in the Python environment. The script does not itself define network architecture, inference thresholds, or metric calculations.

## Configuration

At import/startup, the script locates a workspace named `fruit_detector_ws`, searches beneath `src/aoc_fruit_detector` for a `config` directory, and loads `non_ros_params.yaml`. Missing workspace configuration paths cause `FileNotFoundError`.

Configured file/directory paths beginning with `./` are resolved using the project's `find_path` helper. The following YAML keys are accessed directly:

| Section | Key | Use |
|---|---|---|
| `datasets` | `train_dataset_name` | Training dataset name (read but not directly used here) |
| `datasets` | `test_dataset_name` | Testing dataset name (read but not directly used here) |
| `directories` | `train_image_dir` | Training image directory (read but not directly used here) |
| `directories` | `test_image_dir` | Directory containing test RGB/depth files |
| `directories` | `prediction_json_dir` | JSON output directory |
| `directories` | `prediction_output_dir` | Prediction image output directory |
| `training` | `number_of_classes` | Number of classes (read but not directly used here) |
| `training` | `epochs` | Epoch count (read but not directly used here) |
| `settings` | `download_assets` | Whether to request asset downloads |
| `settings` | `fruit_type` | `STRAWBERRY` or `TOMATO`; otherwise defaults to strawberry |
| `settings.filename_patterns` | `rgb`, `depth` | Substrings for finding and pairing files |

The complete configuration is passed to `DetectronPredictor` and `DetectronTrainer`, which may consume additional settings not shown in this file.

If `settings.download_assets` evaluates to true, `LearnerUtils(config_data).call_download()` is run before test images are listed.

## Processing workflow

1. **Discover images.** List files in `test_image_dir`; select RGB and depth files using their configured filename substrings. Lists are sorted.
2. **Initialise predictor.** Construct `DetectronPredictor(config_data)` inside `call_predictor()`.
3. **Load RGB.** Read each RGB image with `cv2.imread()` (OpenCV BGR channel order).
4. **Match depth.** Replace the RGB substring in the filename with the depth substring and check that the resulting file exists among the discovered depth files.
5. **Prepare depth.** Read matched depth without automatic colour conversion; resize to RGB dimensions using nearest-neighbour interpolation; convert a three-channel depth image to grayscale; expand single-channel depth to `(height, width, 1)`.
6. **Construct RGBD.** Concatenate BGR with depth to obtain four channels. If depth is missing or unreadable, use the RGB image's blue channel as a synthetic fourth channel.
7. **Predict.** Call the Detectron predictor for each input. In regular Python execution, call `get_predictions_image(...)`, including output paths, image filename, image index, and fruit type.
8. **Return annotations.** Yield `json_annotation_message` for each successful prediction; log and report exceptions, then move on to the next image.
9. **Evaluate.** Once the generator has been fully consumed, call `call_trainer(False, True)` for evaluation.

### Input representation

| Input | Handling |
|---|---|
| RGB image | Loaded using `cv2.imread`, producing BGR channels |
| Depth image available | Loaded unchanged, aligned by resizing dimensions, used as channel four |
| Depth image unavailable | Fourth channel duplicates the BGR blue channel |
| `fruit_type` | Converted to `FruitTypes.Strawberry` or `FruitTypes.Tomato` |

The script does not independently validate depth units or perform RGB-depth spatial registration beyond matching image dimensions.

## Functions

### `call_predictor()`

Generator performing file-by-file inference. Constructs the predictor, pairs and preprocesses images, runs inference, and yields the returned annotation message.

**Normal Python mode (`__debug__ == True`):**

```python
json_annotation_message, predicted_image, depth_masks = det_predictor.get_predictions_image(
    rgbd_image,
    prediction_json_output_file,
    prediction_image_output_dir,
    image_file_name,
    sample_no,
    fruit_type,
)
```

**Optimised Python mode (`python -O`, `__debug__ == False`):**

```python
json_annotation_message, predicted_image, rgb_masks, depth_masks = (
    det_predictor.get_predictions_message(rgbd_image, 1, fruit_type)
)
```

The second branch passes a constant dummy image ID of `1`. The script comments indicate that `predicted_image` is `None` in this mode. The exact contents of the returned annotations and mask structures are determined by the external predictor implementation.

**Output:** yields one annotation message per successful image. Once iteration finishes, calls `call_trainer(False, True)`.

### `call_trainer(resumeType=False, skipTraining=False)`

Initialises a `DetectronTrainer`, calls:

```python
aoc_trainer = detTrainer.train_model(
    resumeType=resumeType,
    skipTraining=skipTraining,
)
detTrainer.evaluate_model(aoc_trainer.model)
```

The inference workflow calls this with `skipTraining=True`; intended use is evaluation without training. Actual skip/resume semantics, metrics, and outputs are implemented in `DetectronTrainer`, not in this script.

## Running

From an environment with the project packages installed and the expected workspace/configuration available:

```bash
python predictor.py
```

The main block consumes all values from `call_predictor()` and prints `Prediction generated.` once for each yielded annotation. It prints the number of discovered RGB and depth files before running inference.

For the alternative optimised branch:

```bash
python -O predictor.py
```

**Caution:** Optimised mode selects a different predictor method with a different expected return signature; test it against the installed predictor implementation rather than assuming both modes behave identically.

## Outputs

- Annotation messages returned by the predictor and yielded by `call_predictor()`.
- A JSON output file path built from each RGB filename when `prediction_json_dir` is nonempty.
- Prediction image output location passed through `prediction_output_dir`.
- Evaluation initiated after all test images are processed.
- Console/log output for discovered file counts, successes, and errors.

The actual JSON schema, whether directories are created, output image formats, and evaluation metric files are not defined in `predictor.py` and require inspecting the predictor/trainer implementations.

## Implementation notes and potential issues

1. **Unreadable RGB images:** A failed `cv2.imread()` is logged but not skipped. Later access to `rgb_image.shape` or `rgb_image[:, :, 0]` may raise an exception before the inference `try` block.
2. **Blue-channel fallback:** Missing depth does not result in real depth estimation; the fourth channel is copied from BGR blue. Predictions may differ from models trained on genuine depth.
3. **Depth alignment:** Nearest-neighbour resizing matches image dimensions but does not ensure camera-calibrated alignment.
4. **Depth formats:** A single-channel or three-channel depth array is supported by the preparation logic; other channel layouts are not explicitly handled.
5. **No output directory creation:** The script constructs output paths without calling `os.makedirs()` for the JSON/image directories.
6. **Evaluation depends on generator exhaustion:** `call_trainer(False, True)` runs only after the entire `call_predictor()` generator is consumed. Merely creating the generator does not run inference or evaluation.
7. **File pairing:** Substring-based `str.replace()` may replace more than one occurrence, and naming collisions are not checked.
8. **Prediction exceptions:** Errors inside the prediction `try` block are logged and processing continues; exceptions during earlier image/depth preparation are outside that handler.
9. **At import time:** Workspace lookup, configuration loading, optional downloads and directory listing run even if the module is imported rather than executed as a script.

## Scope

This document describes the supplied `predictor.py` file. It does not describe a ROS node, model training internals, performance metrics, or segmentation architecture beyond the calls explicitly present in the source.
