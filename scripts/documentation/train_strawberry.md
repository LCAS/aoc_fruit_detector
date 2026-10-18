# `train_strawberry.py` — Strawberry ripeness training

## Purpose

`train_strawberry.py` trains a two-class Mask R-CNN instance segmentation model for ripe and unripe strawberries. It prepares detector-compatible COCO annotations, registers training/validation/test datasets through `DetectronTrainer`, and runs a `StrawberryTrainer` subclass of `AOCTrainer`.

The script adds training/validation loss plots and selects `model_best.pth` using **ripe-only validation segmentation AP**, `segm/AP-fruit_ripe`. After training, it loads that checkpoint and evaluates it on the held-out test split.

## Dependencies

- Detectron2: model zoo, dataset mappers/loaders, checkpointing, training hooks and event storage.
- Local project: `DetectronTrainer` and `AOCTrainer`.
- Python packages: `torch`, `numpy`, `matplotlib`, `PyYAML`.
- Standard library: `json`, `os`, `pathlib`, `contextlib`.

Matplotlib uses the non-interactive `Agg` backend, so graph generation does not require a desktop display. The pretrained model must be cached or downloadable. Device selection is inherited from the trainer configuration; this script does not implement automatic CPU fallback.

## Running

From the workspace root, with the project dependencies available:

```bash
python3 src/scripts/train_strawberry.py
```

The script has no command-line options. Dataset, configuration and output paths are resolved relative to the script, so they do not depend on the initial working directory.

Training starts with `resume_or_load(resume=False)`: it loads the configured COCO pretrained weights rather than resuming an earlier run. Outputs use a fixed directory; preserve previous run artifacts before starting another experiment if they are needed for comparison.

## Dataset layout and preparation

Expected layout under `src/scripts/`:

```text
data/strawberry/
├── train/
│   ├── annotations/ripeness_class_annotations.json
│   └── <images referenced by the annotations>
├── val/
│   ├── annotations/ripeness_class_annotations.json
│   └── <images referenced by the annotations>
└── test/
    ├── annotations/ripeness_class_annotations.json
    └── <images referenced by the annotations>
```

Each annotation file must use COCO format with `images`, `annotations` and `categories`. Image filenames are resolved against their split directory, including any relative subdirectories.

`prepare_datasets()` normalizes category names and IDs:

| Accepted source name | Exported category ID | Exported name |
|---|---|---|
| `ripe`, `fruit_ripe` | `1` | `fruit_ripe` |
| `unripe`, `fruit_unripe` | `2` | `fruit_unripe` |

Both classes must be declared in each split's categories. Unknown category names fail preparation. Images whose files do not exist are omitted, along with their annotations; the script prints the retained and omitted image counts. Source images and annotation files are not modified. Cleaned annotations are written to the training output directory.

## Configuration

The script reads `src/config/non_ros_params.yaml`, then overrides dataset names, annotation/image paths, model architecture, pretrained weights, output paths and the settings below. Editing those same keys only in the source YAML will not override the values assigned by `main()`.

| Setting | Current value | Meaning |
|---|---|---|
| Architecture | `COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml` | ResNet-101 FPN Mask R-CNN |
| Initial weights | Model-zoo COCO checkpoint for that architecture | Training initialization |
| Classes | `2` | Ripe and unripe |
| `epochs` | `10000` | Maximum training **iterations**, not dataset epochs |
| Learning rate | `0.00025` | Base learning rate |
| Batch size | `2` | Images per training batch |
| Workers | `2` | Data-loader workers |
| Learning-rate steps | `[2000, 2600]` | Iterations at which the inherited scheduler decays the rate |
| Warmup iterations | `100` | Initial learning-rate warmup |
| Validation period | `500` | Validation evaluation and graph refresh interval |
| Asset download helper | Disabled | Does not disable pretrained checkpoint downloading by Detectron2 |

The optimizer choice is inherited from `training.optimizer` in the source YAML. `training_directory()` temporarily changes the working directory to the output folder, where a copy of the configuration is placed at `data/config/config.yaml` for `AOCTrainer.build_optimizer()`. The previous working directory is restored afterward.

`training_config.yaml` records the configuration dictionary, not a complete dump of the final Detectron2 configuration. Its `model_file` points to the initialization checkpoint, not the selected trained model.

## Processing sequence

1. Prepare normalized train, validation and test annotation files.
2. Read the source YAML and apply run-specific overrides.
3. Save the training configuration and initialize `DetectronTrainer`, which registers datasets and exports catalogs.
4. Apply batch size, worker count, output directory, learning-rate steps and warmup to the Detectron2 configuration.
5. Instantiate `StrawberryTrainer`, load pretrained weights without resuming, and train.
6. Run validation evaluation, loss measurement and best-checkpoint selection at the configured interval.
7. Check that `model_best.pth` exists and print the inference configuration paths.
8. Load `model_best.pth` into the model and call `DetectronTrainer.evaluate_model()` on the test dataset.

## Augmentation

Training augmentation is inherited from `AOCTrainer.build_train_augmentation()`:

| Transform | Current behavior |
|---|---|
| Resize | Shortest-edge resize using the model configuration |
| Crop | Conditional on `cfg.INPUT.CROP.ENABLED` |
| Vertical flip | Probability `0.5` |
| Horizontal flip | Probability `0.5` |
| Contrast | Random factor from `0.8` to `3.0` |
| Brightness | Random factor from `0.3` to `1.6` |

The separate validation-loss mapper uses only shortest-edge resizing with the test input sizes. It uses `is_train=True` to retain annotations needed to calculate losses, but does not apply the training flips or lighting transforms.

## Validation and best-model selection

During training, `cfg.DATASETS.TEST` refers to **the validation split**, registered as `strawberry_val`. The held-out test split is evaluated only after training.

`AOCTrainer` supplies the COCO evaluator. The additional `BestCheckpointer` in this script monitors `segm/AP-fruit_ripe` in maximum mode and saves the best checkpoint as `model_best.pth`.

This metric measures ripe mask detection AP across COCO IoU thresholds from 0.50 to 0.95. It is not validation loss, overall two-class AP, confidence or ripe recall. Both classes still participate in training and evaluation; the checkpoint selection criterion alone prioritizes ripe performance.

## Loss graph and history

`LossGraphHook` records training total loss after each iteration. At validation intervals and the final iteration, it evaluates mean total loss over the full validation loader without gradients. It temporarily enables model training mode to obtain loss dictionaries while keeping batch normalization and dropout in evaluation mode, then restores each module's original mode.

Validation runs inside isolated PyTorch RNG and event-storage contexts. Empty validation data or non-finite validation loss raises an error.

The hook writes:

- `loss_curve.png`: training total loss and validation total loss versus iteration.
- `loss_history.json`: `training` and `validation` lists of `[iteration, loss]` pairs.
- `validation_loss`: a scalar in Detectron2's event storage.

The graph does **not** plot ripe AP and does not choose the best checkpoint. Per-class AP is recorded by the evaluator in the training metrics. A lower total loss need not correspond to higher ripe AP. Graph/history files are refreshed using temporary files followed by replacement.

## Output files

Outputs are stored in `src/scripts/data/training_output/strawberry_ripeness/`.

| Artifact | Purpose |
|---|---|
| `train_annotations.json`, `val_annotations.json`, `test_annotations.json` | Normalized annotation copies |
| `training_config.yaml` | Run configuration snapshot |
| `data/config/config.yaml` | Configuration read by the inherited optimizer |
| `train_dataset_catalog.pkl` | Serialized training dataset records |
| `test_metadata_catalog.pkl` | Serialized metadata and normalized category definitions |
| `model_best.pth` | Checkpoint selected by ripe validation segmentation AP |
| `model_final.pth` | Final checkpoint from the inherited training loop |
| `metrics.json`, TensorBoard event files | Training/evaluation scalars from Detectron2 writers |
| `loss_curve.png`, `loss_history.json` | Loss graph and its underlying samples |
| Evaluation prediction files | COCO evaluator outputs, including validation outputs under `inference/` |

## Using the trained model

Set these values in `src/config/non_ros_params.yaml`, replacing the output prefix with the absolute path on the target machine:

```yaml
files:
  model_file: /absolute/path/to/strawberry_ripeness/model_best.pth
  test_metadata_catalog_file: /absolute/path/to/strawberry_ripeness/test_metadata_catalog.pkl
  config_file: COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml
training:
  number_of_classes: 2
```

Merge these keys into the existing YAML rather than replacing its other required settings. The script prints these paths but does not automatically update the inference YAML. The inference confidence threshold remains a separate setting.

## Implementation caveats

- **Early learning-rate decay:** the current run lasts 10,000 iterations but still decays at 2,000 and 2,600. Increasing `epochs` alone does not move these steps.
- **Inherited SGD behavior:** the current custom optimizer omits `momentum`, so PyTorch's default of zero applies even if `cfg.SOLVER.MOMENTUM` is nonzero.
- **Evaluation cutoff:** the wrapper sets `MODEL.ROI_HEADS.SCORE_THRESH_TEST` to `0.5`; this filters candidates before AP evaluation. The inference YAML confidence threshold is not the checkpoint selection criterion.
- **Fixed output location:** repeated runs can overwrite artifacts or leave older files in the directory. Loss history starts empty on each run; it is not restored from disk.
- **Validation versus test:** choose settings using validation results; the final test evaluation is a separate assessment of the selected checkpoint.

## Main functions and classes

| Symbol | Responsibility |
|---|---|
| `prepare_datasets()` | Normalize category IDs/names and remove missing-image records |
| `build_training_hooks()` | Build validation-loss and ripe-AP checkpoint hooks |
| `LossGraphHook.validation_loss()` | Compute held-out mean total loss |
| `LossGraphHook.after_step()` | Record training loss and trigger validation/plotting |
| `LossGraphHook.save_graph()` | Save loss graph and JSON history |
| `StrawberryTrainer.build_hooks()` | Extend inherited trainer hooks |
| `training_directory()` | Provide the optimizer's expected configuration layout |
| `main()` | Prepare, configure, train, select and test the model |

## Related modules

- [detectron_trainer.md](detectron_trainer.md): configuration, dataset registration and test evaluation.
- [aoc_trainer.md](aoc_trainer.md): inherited augmentation, optimizer, scheduler and evaluator.
- [detectron_predictor.md](detectron_predictor.md): inference and model loading.
- [fruit_detection.md](fruit_detection.md): ROS detection pipeline and runtime configuration.
