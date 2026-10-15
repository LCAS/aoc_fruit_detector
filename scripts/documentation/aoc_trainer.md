# AOCTrainer (`aoc_trainer.py`)

## Purpose

`AOCTrainer` extends Detectron2's `DefaultTrainer` to customise training-time image augmentation, optimizer selection, data loading, learning-rate scheduling and COCO-format evaluation. The module defines a trainer class; it does **not** start training itself.

## Dependencies

- Python: `copy`, `os`, `yaml`
- PyTorch: `torch`
- OpenCV: `cv2`
- Detectron2: `DefaultTrainer`, `DatasetMapper`, `build_detection_train_loader`, transformation utilities, optimizer utilities and `COCOEvaluator`
- Detectron2 DeepLab project: `build_lr_scheduler`

## Class overview

| Method | Function |
| --- | --- |
| `build_train_augmentation(self, cfg)` | Creates the training augmentation sequence. |
| `build_optimizer(cls, cfg, model)` | Selects Adam or SGD and prepares trainable parameters. |
| `build_train_loader(cls, cfg)` | Builds the Detectron2 training data loader with the custom augmentation list. |
| `build_lr_scheduler(cls, cfg, optimizer)` | Delegates scheduler construction to Detectron2's DeepLab scheduler function. |
| `custom_mapper(cls, dataset_dict)` | Experimental mapping routine for image and annotation transforms. |
| `hsv_convert(cls, dataset_dict)` | Experimental image conversion routine; actually uses **YUV**, not HSV. |
| `build_evaluator(cls, cfg, dataset_name, output_folder=None)` | Constructs a `COCOEvaluator` for a named dataset. |

## Training augmentation

`build_train_augmentation()` constructs these transforms in order:

1. `ResizeShortestEdge` using `cfg.INPUT.MIN_SIZE_TRAIN`, `MAX_SIZE_TRAIN` and `MIN_SIZE_TRAIN_SAMPLING`.
2. Optional `RandomCrop_CategoryAreaConstraint` when `cfg.INPUT.CROP.ENABLED` is true, using the crop settings and `cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE`.
3. Vertical random flip (`prob=0.5`).
4. Horizontal random flip (`prob=0.5`).
5. Random contrast (`0.8` to `3`).
6. Random brightness (`0.3` to `1.6`).

The two flips are applied independently, so both can occur on the same image. The configured contrast and brightness ranges can produce relatively strong appearance changes.

`build_train_loader()` supplies this list to `DatasetMapper(cfg, is_train=True, augmentations=...)` and passes the mapper to `build_detection_train_loader`.

## Optimizer configuration

`build_optimizer()` first gathers parameters using Detectron2's `get_default_optimizer_params`, including the configured learning-rate and bias/normalisation weight-decay options. It then reads:

```text
./data/config/config.yaml
```

Relative paths are resolved from the **process working directory**, not necessarily the location of `aoc_trainer.py`.

Expected YAML setting:

```yaml
training:
  optimizer: ADAM  # or SGD
```

- `ADAM` (case-insensitive) selects `torch.optim.Adam`.
- Any other optimizer string selects `torch.optim.SGD`.
- If loading or interpreting the YAML fails, the method silently falls back to SGD (the exception is caught).
- Both optimizers receive `cfg.SOLVER.BASE_LR` and `cfg.SOLVER.WEIGHT_DECAY` and are passed through `maybe_add_gradient_clipping`, so clipping depends on Detectron2 configuration.
- Neither optimizer's other hyperparameters (such as momentum or Adam betas) are explicitly set in this method.

## Learning-rate schedule

`build_lr_scheduler()` calls `detectron2.projects.deeplab.build_lr_scheduler(cfg, optimizer)`. The resulting schedule therefore depends on the supplied Detectron2 configuration and the DeepLab scheduler implementation; no schedule values are hard-coded in this file.

## Additional mapping utilities

### `custom_mapper(dataset_dict)`

Intended to create training examples by reading BGR images, resizing to `400 × 400`, randomly modifying brightness and flipping images, transforming the corresponding instance annotations, and converting images into `float32` CHW tensors. It excludes annotations where `iscrowd != 0` and filters empty instances.

**Important:** This routine is not passed to `build_train_loader()` and has implementation issues noted below. It should not be assumed to be active in training.

### `hsv_convert(dataset_dict)`

Reads images with OpenCV, converts them using `cv2.COLOR_BGR2YUV`, and converts the result to a `float32` CHW tensor. Despite its name and comment, it performs **YUV**, not HSV conversion. It is not wired into `build_train_loader()`.

## Evaluation

`build_evaluator()` returns:

```python
COCOEvaluator(dataset_name, cfg, True, output_folder)
```

If no output directory is passed, it uses `<cfg.OUTPUT_DIR>/inference`. Metrics are determined by Detectron2's COCO evaluator and the dataset's annotations/tasks; this module does not define a custom metric.

## Configuration dependencies

| Setting | Use |
| --- | --- |
| `INPUT.MIN_SIZE_TRAIN`, `INPUT.MAX_SIZE_TRAIN`, `INPUT.MIN_SIZE_TRAIN_SAMPLING` | Resize configuration |
| `INPUT.CROP.ENABLED`, `TYPE`, `SIZE`, `SINGLE_CATEGORY_MAX_AREA` | Optional crop configuration |
| `MODEL.SEM_SEG_HEAD.IGNORE_VALUE` | Crop category-area constraint |
| `SOLVER.BASE_LR`, `WEIGHT_DECAY`, `WEIGHT_DECAY_NORM`, `BIAS_LR_FACTOR`, `WEIGHT_DECAY_BIAS` | Optimizer parameter settings |
| Gradient-clipping and scheduler configuration | Applied by Detectron2 helper functions |
| `OUTPUT_DIR` | Default evaluation output location |
| `./data/config/config.yaml`: `training.optimizer` | Adam versus SGD selection |

## Implementation observations and limitations

1. **Augmentation method signature:** `build_train_augmentation` is decorated with `@classmethod` but declares `(self, cfg)`. Called from `cls.build_train_loader(cfg)`, `self` receives the class, so this works, although `cls` would be the clearer name.
2. **`custom_mapper` does not return processed records:** It constructs `dataset_dicts` but returns the original `dataset_dict`. As written, modifications may not be reflected in the return value.
3. **`custom_mapper` expects an iterable of records:** It loops over `dataset_dicts`, whereas Detectron2's `DatasetMapper` normally receives a single dataset record. The routine is not connected to the loader.
4. **`hsv_convert` assumes an iterable of records** and is not connected to the loader. It also does not transform annotations, and its name misidentifies the colour space.
5. **Optimizer errors are swallowed:** A missing/invalid YAML file or other exception in optimizer selection will select SGD without logging the cause.
6. **Evaluation is COCO-based:** A correctly registered dataset and compatible annotations are required; neither registration nor dataset configuration is implemented here.
7. **Training execution lives elsewhere:** Checkpoints, epochs, dataset registration, and training invocation are not specified by this class alone.

## Typical integration

Once a Detectron2 `cfg` and training/evaluation datasets have been configured elsewhere, the class can be used in the standard `DefaultTrainer` pattern:

```python
trainer = AOCTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

This is an integration example, **not** an entry point present in the uploaded source file.
