# Detectron Predictor (`detectron_predictor.py`)

## Purpose

`DetectronPredictor` is an inference wrapper around Detectron2's `DefaultPredictor` for detecting and segmenting fruit in images. It converts model instances to project-specific JSON annotations using `JSONWriter`, renders predictions using `AOCVisualizer`, and optionally derives class-wise depth masks from a supplied RGB-D image.

**Important:** The Detectron2 model receives **only the first three image channels (RGB/BGR)**. The fourth channel, if supplied, is used *after inference* to calculate depth masks; it is not model input.

## Dependencies

- Python: `numpy`, `pickle`, `os`, `logging`, `traceback`, `datetime`
- Third-party: `detectron2`, `opencv-python` (`cv2`)
- Project: `LearnerPredictor`, `LearnerUtils`, `AOCVisualizer`, `ColorMode`, `JSONWriter`, `FruitTypes`

The class inherits from `LearnerPredictor`. It is a library class, not a standalone command-line program (there is no `main()` in this file).

## Configuration

The constructor accepts a dictionary `config_data`, plus optional `scale=1.0` and `instance_mode=ColorMode.SEGMENTATION`.

| Config key | Purpose |
|---|---|
| `files.model_file` | Saved model weights path |
| `files.config_file` | Detectron2 model-zoo configuration name/path used by `model_zoo.get_config_file()` |
| `files.test_metadata_catalog_file` | Pickled metadata and category information |
| `files.train_dataset_catalog_file` | Stored dataset catalog path (not subsequently used in this file) |
| `training.number_of_classes` | Sets `cfg.MODEL.ROI_HEADS.NUM_CLASSES` |
| `training.epochs` | Loaded but not used for inference here |
| `settings.download_assets` | Invoke `LearnerUtils.call_download()` during initialization |
| `settings.rename_pred_images` | Rename output images/JSON to sequential `img_000001` etc. |
| `settings.show_orientation` | Show orientation in image-based visualization methods |
| `settings.bbox` | Bounding-box visualization setting |
| `settings.segm_masks` | Segmentation visualization setting |
| `settings.confidence_threshold` | Minimum confidence threshold, clamped to `[0,1]` |

The actual filtering condition is **`score > confidence_threshold`**, not `>=`.

### Initialization

1. Read configuration options and optionally download assets.
2. Unpickle the test metadata catalog with `_get_catalog()`.
3. Build Detectron2 configuration with `_configure()`; load model-zoo config, set weights and class count.
4. Instantiate `DefaultPredictor(self.cfg)`.

`_get_catalog()` expects metadata in `data[0]` and category entries in `data[-1]`. It reverses `thing_colors` and converts each category ID to a zero-based index (`id - 1`). The image methods use `self.metadata[1]` as categories, so the catalog format must also support that indexing.

## Inference flow

```text
Image (RGB or RGB-D)
  -> first three channels as uint8
  -> Detectron2 DefaultPredictor
  -> instances moved to CPU
  -> strict confidence filtering
  -> JSONWriter -> prediction annotations
  -> optional AOCVisualizer -> rendered image
  -> optional colour-based class masks + depth masks
  -> optional saved image and JSON
```

The output JSON schema, fruit orientation computation, and visualization styling are defined by the imported `JSONWriter` and `AOCVisualizer` components, not by this script.

## Public prediction methods

| Method | Input | Saves output | Returns |
|---|---|---|---|
| `get_rgb_predictions_image(...)` | Three-channel image | Overlay in `predicted_images/` and JSON | `(predicted_json_ann, predicted_image, [])` |
| `get_predictions_image(...)` | Four-channel RGB-D image | Overlay in `predicted_images/` and JSON | `(predicted_json_ann, predicted_image, depth_masks)` |
| `get_predictions_message(...)` | Four-channel RGB-D image | No image/JSON file | `(predicted_json_ann, predicted_image, rgb_masks, depth_masks)` |
| `get_predictions_message_short(...)` | Image with at least three channels | No image/JSON file | `predicted_json_ann` |

### `get_rgb_predictions_image`

Runs prediction on the first three channels, filters by confidence, draws the selected detections, saves the rendered output and calls `JSONWriter.create_prediction_json(..., save_json_file=True)`. It returns an empty list as the third item because depth masks are not calculated.

### `get_predictions_image`

Expects channel index `3` to hold depth. Runs the same first-three-channel Detectron2 prediction, generates a visualization, extracts `depth_masks` using `get_masks()`, and writes overlay/JSON outputs. The calculated `rgb_masks` are not included in its return value.

### `get_predictions_message`

Designed for in-memory integration (for example, a ROS node). It runs prediction, builds an overlay, obtains class-wise RGB and depth masks, and returns JSON-like annotations without writing files (`save_json_file=False`). It explicitly sets `show_orientation=False` for the visualizer regardless of the configuration value.

### `get_predictions_message_short`

Runs detection and JSON conversion **without visualization or depth-mask calculation**. Only the first three channels are read. This is the most direct method when only the annotation payload is needed; no performance figures are provided by this source.

## Saved outputs

For the two image-saving methods:

- Annotated images: `<prediction_output_dir>/predicted_images/`.
- JSON: `output_json_file_path` supplied to `JSONWriter` (or a renamed `img_000001.json` path).
- With `rename_pred_images=True`, image and JSON names use the six-digit `sample_no`.
- With renaming disabled, the overlay uses the source `image_file_name` basename.
- The reported `Duration` measures the `draw_instance_predictions()` visualization call, **not** complete inference runtime.

The caller must provide appropriate writable output locations; the script itself creates the annotated-image directory, but does not explicitly create the parent directory for the JSON path.

## Mask calculation: `get_masks(fg_masks, rgb_image, depth_image)`

Despite the `fg_masks` name, the function receives a **colour-rendered prediction image**, rather than Detectron2's original per-instance binary masks.

For each configured `(colour, category_id)` pair, it:

1. Finds pixels where all three colour channels exactly equal the configured class colour.
2. Multiplies this binary selection by `depth_image` to obtain a class-wise depth mask.
3. Multiplies the binary selection by `category_id + 1` to encode the segmentation class.
4. Stacks masks along the third dimension when multiple classes exist.

Returns `(classwise_depth_masks, classwise_segm_masks)`. The second return value is called `rgb_masks` in the prediction methods, although it actually holds category-coded segmentation masks, **not RGB imagery**.

**Limitation:** Colour matching depends on rendered output pixels and colours. Overlay effects, annotations, blending, or colour-channel conventions can cause these masks to differ from the model's underlying segmentation masks. The comment mentions deriving a background mask, but the implemented function does **not** explicitly calculate one.

## Usage example

```python
from detectron_predictor.detectron_predictor import DetectronPredictor

# config_data must contain the configuration keys listed above.
model = DetectronPredictor(config_data)
annotations = model.get_predictions_message_short(image, image_id=1)
```

`image` should contain at least three image channels. For the full message or image RGB-D methods, supply a fourth depth channel.

## Implementation notes / limitations

- **RGB-only network inference:** No depth channel is passed into `DefaultPredictor`.
- **Strict threshold:** Detections whose score equals the threshold are excluded.
- **Default image method arguments:** Calling image-saving methods without valid output directories/paths is not guaranteed to succeed; they attempt disk writes.
- **Metadata assumptions:** The pickled catalog must have the expected indexed entries, `thing_colors`, and category IDs.
- **Mask fragility:** Class masks are reconstructed from *visualized colours*, not native Detectron2 instance mask tensors.
- **No explicit device override:** Device selection is determined by the loaded Detectron2 configuration/environment.
- **No inference-time statistics:** Only the time spent drawing predictions is measured in the two image-saving methods.
- **Unused inputs/settings:** `rgb_image` is not used inside `get_masks()`. `epochs` and `dataset_catalog_file` are loaded but not used elsewhere in this file.
- **Potential empty-catalog issue:** `get_masks()` assumes at least one class colour/category pair; otherwise the output variables are never initialized.

