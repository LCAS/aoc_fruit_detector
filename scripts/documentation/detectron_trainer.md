# `detectron_trainer.py` — Detectron2 training and evaluation

## Purpose

`DetectronTrainer` is a project-specific wrapper around Detectron2's training and COCO evaluation facilities. It reads dataset, model and training settings from a supplied `config_data` dictionary, registers COCO-format training, validation and test datasets, constructs an `AOCTrainer`, and evaluates its model on the held-out test dataset.

The class inherits from `LearnerTrainer`; the actual training loop and training-time augmentation are delegated to `AOCTrainer` (defined in `detectron_trainer.aoc_trainer`). The file does **not** define an executable `main` function.

## Dependencies

- Detectron2: `get_cfg`, `model_zoo`, `register_coco_instances`, `DatasetCatalog`, `MetadataCatalog`, `DefaultTrainer`, `COCOEvaluator`, `build_detection_test_loader`, `inference_on_dataset`.
- Local project: `AOCTrainer`, `LearnerTrainer`, `LearnerUtils`.
- Python standard library: `os`, `pickle`, `json`, `logging`, `traceback`.

The script sets the `detectron2` logger level to `WARNING`.

## Configuration input

The constructor receives a parsed dictionary (for example, one read from a YAML configuration file). The following entries are accessed:

| Section | Key | Purpose |
|---|---|---|
| `datasets` | `train_dataset_name` | Registered training dataset name |
| `datasets` | `test_dataset_name` | Registered held-out test dataset name |
| `datasets` | `validation_dataset_name` | Registered validation dataset name |
| `files` | `model_file` | Weights path used by `_configure()` under a condition |
| `files` | `config_file` | Detectron2 model-zoo configuration identifier |
| `files` | `train_annotation_file` | Training COCO annotations |
| `files` | `test_annotation_file` | Test COCO annotations |
| `files` | `validation_annotation_file` | Validation COCO annotations |
| `files` | `train_dataset_catalog_file` | Destination for pickled training dataset records |
| `files` | `test_metadata_catalog_file` | Destination for pickled test metadata and categories |
| `files` | `pretrained_model_file` | Controls whether weights are assigned in `_configure()` |
| `training` | `number_of_classes` | Number of detection classes |
| `training` | `epochs` | Passed to Detectron2 as maximum **iterations** |
| `training` | `learning_rate` | Base learning rate |
| `settings` | `validation_period` | Iteration interval for validation evaluation |
| `settings` | `download_assets` | Whether to invoke asset downloading |
| `directories` | `train_image_dir` | Training image directory |
| `directories` | `test_image_dir` | Test image directory |
| `directories` | `validation_image_dir` | Validation image directory |

If `download_assets` is true, `LearnerUtils(config_data).call_download()` runs before configuration and dataset registration.

## Initialization and processing sequence

1. `__init__(config_data)` stores settings and optionally downloads assets.
2. `_configure(self.epochs, self.learning_rate)` creates the Detectron2 configuration.
3. `_register_train_dataset()` registers the training COCO dataset and serializes its dataset records.
4. `_register_test_dataset()` registers the test COCO dataset and serializes metadata plus categories.
5. `_register_validation_dataset()` registers the validation COCO dataset.
6. A caller separately invokes `train_model()` and, when required, `evaluate_model()`.

## Detectron2 configuration

`_configure(iterations=10000, learning_rate=0.0025, num_workers=8, batch_size=8, batch_per_image=512, test_threshold=0.5)` starts with the named Detectron2 model-zoo configuration and overrides:

| Detectron2 setting | Assigned value |
|---|---|
| `MODEL.ROI_HEADS.NUM_CLASSES` | `number_of_classes` |
| `DATASETS.TRAIN` | Training dataset tuple |
| `DATASETS.TEST` | **Validation** dataset tuple during training |
| `TEST.EVAL_PERIOD` | `validation_period` |
| `DATALOADER.NUM_WORKERS` | `8` by default |
| `SOLVER.IMS_PER_BATCH` | `8` by default |
| `SOLVER.BASE_LR` | `learning_rate` |
| `SOLVER.MAX_ITER` | Constructor's `epochs` value |
| `MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE` | `512` by default |
| `MODEL.ROI_HEADS.SCORE_THRESH_TEST` | `0.5` by default |

When `pretrained_model_file` is nonempty, the code sets `cfg.MODEL.WEIGHTS` to `model_file`, **not** `pretrained_model_file`.

## Dataset registration

All three datasets use `register_coco_instances(name, {}, annotation_file, image_dir)`.

- **Training:** calls `DatasetCatalog.get(train_name)` and pickles its dataset records to `train_dataset_catalog_file`.
- **Test:** sets metadata `thing_colors` to green and red, registers the dataset, then reads `categories` from the **training** annotation JSON and pickles `[test_metadata, categories]` to `test_metadata_catalog_file`.
- **Validation:** sets the same two metadata colors and registers the dataset; no catalog file is written here.

The specified color list contains two colors regardless of the configured class count.

## Training: `train_model(resumeType=False, skipTraining=False)`

Creates `cfg.OUTPUT_DIR`, instantiates `AOCTrainer(cfg)`, calls `resume_or_load(resume=resumeType)`, and then calls `train()` unless `skipTraining=True`. Returns the `AOCTrainer` instance.

- `resumeType=True` requests Detectron2 checkpoint resumption.
- `skipTraining=True` loads/resumes the trainer without running the training loop.
- Augmentations, optimizer choice and the scheduler come from `AOCTrainer` and its configuration, not this file.

## Test evaluation: `evaluate_model(trainer)`

Constructs a `COCOEvaluator` for `test_dataset_name`, builds a test loader for that dataset, and calls:

```python
inference_on_dataset(trainer, val_loader, evaluator)
```

The result is printed, not returned or explicitly written to a separate report by this method. Despite its local variable name `val_loader`, this loader uses the **test** dataset.

**Important:** The argument passed to `evaluate_model` needs to be a callable inference model compatible with `inference_on_dataset` (typically `aoc_trainer.model`), not the `AOCTrainer` wrapper itself.

## Example usage

```python
trainer_wrapper = DetectronTrainer(config_data)
aoc_trainer = trainer_wrapper.train_model(resumeType=False, skipTraining=False)
trainer_wrapper.evaluate_model(aoc_trainer.model)
```

`config_data` must already have the structure listed above, and annotation/image/model paths must exist or be obtainable through the asset downloader.

## Notable implementation caveats

1. **`epochs` is misleading:** it is assigned to `SOLVER.MAX_ITER`, so it specifies iterations, not full dataset epochs.
2. **Weights path mismatch:** checking `pretrained_model_file` but loading `model_file` could select unintended weights; verify which path is supposed to be used.
3. **Categories source:** test metadata is paired with categories read from the training annotations; train and test categories must be consistent.
4. **Limited configurability:** worker count, image batch size, ROI batch size, and test score threshold are hard-coded defaults in `_configure()` and are not read from `config_data`.
5. **Catalog pickling:** persisted Detectron2 metadata/catalog objects and dataset records may be tied to the generating environment and should be handled cautiously.
6. **Two fixed colors:** color metadata assumes two display colors even if `number_of_classes` differs.
7. **Exception handling:** errors are logged and re-raised using `raise Exception(e)`, which may obscure the original exception type.
8. **Evaluation argument:** passing `aoc_trainer` rather than `aoc_trainer.model` to `evaluate_model()` is not the intended usage.

## Related modules

- `aoc_trainer.py`: customized Detectron2 `DefaultTrainer` with augmentation, optimizer, training loader, scheduler, and evaluator configuration.
- `detectron_predictor.py`: loads a configured model for inference and creates image/JSON outputs.
