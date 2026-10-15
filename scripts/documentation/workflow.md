# Fruit Detection Workflow

1. **Set configuration** — Edit `non_ros_params.yaml`: dataset paths, model configuration/weights, number of classes, training settings, output directories and fruit type. For training, `AOCTrainer` also reads optimizer choice from `./data/config/config.yaml`.
2. **Prepare datasets** — Provide COCO-format train, validation and test annotations/images; optionally enable asset downloads (`download_assets`).
3. **Train** — Use `DetectronTrainer.train_model()` (via `predictor.py` → `call_trainer()` or another caller). It registers datasets and trains through `AOCTrainer` with augmentations.
4. **Evaluate** — Run `DetectronTrainer.evaluate_model(trainer.model)` on the test set using `COCOEvaluator`. The supplied `predictor.py` calls evaluation after its prediction loop, with training skipped.
5. **Predict offline** — Run `predictor.py` using images from `test_image_dir`; `DetectronPredictor` produces predictions, visualization images and JSON outputs. Depth images are optional.
6. **Predict via ROS 2** — Start `fruit_detection.py` with `--config-file <path>` and ROS parameter `use_ros:=true`; subscribe to RGB/depth/camera-info topics and publish `fruit_info`, with optional composed images and 3D markers.

**Note:** `predictor.py` primarily runs inference, not training. Its final `call_trainer(False, True)` skips training and performs evaluation. The ROS node's 3D pose estimation requires valid depth and camera calibration (or uses configured fallbacks).
