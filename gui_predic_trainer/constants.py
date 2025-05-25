# ============================================================
# File: constants.py
# Author: Yael Vicente
# Date: 2025-May-06
# Description:
#   Dictionary containing formal English descriptions for YAML configuration
#   fields used in the AOC Fruit Detector GUI. These descriptions are
#   displayed as tooltips to assist the user during configuration.
# ============================================================

FIELD_DESCRIPTIONS = {

    # ROS-specific parameters (aoc_fruit_detector.ros__parameters.*)
    "aoc_fruit_detector.ros__parameters.min_depth": "Minimum depth threshold (in meters) for considering objects during detection.",
    "aoc_fruit_detector.ros__parameters.max_depth": "Maximum depth threshold (in meters) to validate detected objects.",
    "aoc_fruit_detector.ros__parameters.constant_depth_value": "Fixed depth value to be used when depth data is not available.",
    "aoc_fruit_detector.ros__parameters.fruit_type": "Specifies the type of fruit to detect within the ROS pipeline. Supported values: 'strawberry', 'tomato'.",
    "aoc_fruit_detector.ros__parameters.pose3d_frame": "TF frame used to publish the 3D pose of detected fruits.",
    "aoc_fruit_detector.ros__parameters.pose3d_tf": "Determines whether pose transformation between frames is required (True = apply transformation).",
    "aoc_fruit_detector.ros__parameters.verbose": "List of boolean flags to enable visual annotations: [centroid, bounding box, mask, coordinate frame, text].",
    "aoc_fruit_detector.ros__parameters.pub_verbose": "Enables publishing of annotated prediction images on a ROS topic.",
    "aoc_fruit_detector.ros__parameters.pub_markers": "Enables publishing of RViz markers for 3D fruit pose visualization.",
    "aoc_fruit_detector.ros__parameters.use_ros": "If enabled, the model will run within the ROS2 framework using topics for I/O.",

    # Dataset-related fields
    "datasets.train_dataset_name": "Name of the training dataset. Use the same name as the dataset used to train the selected model. Required.",
    "datasets.test_dataset_name": "Name of the testing dataset. Use the same name as the dataset used to train the selected model. Required.",
    "datasets.validation_dataset_name": "Name of the validation dataset. Use the same name as the dataset used to train the selected model. Required.",
    "datasets.dataset_train_annotation_url": "URL for the training dataset's annotation file. Required if 'download_assets' is enabled.",
    "datasets.dataset_train_images_url": "URL for the training dataset's image folder. Required if 'download_assets' is enabled.",
    "datasets.dataset_test_annotation_url": "URL for the test dataset's annotation file. Required if 'download_assets' is enabled.",
    "datasets.dataset_test_images_url": "URL for the test dataset's image folder. Required if 'download_assets' is enabled.",

    # File paths
    "files.pretrained_model_file": "Path to a pretrained model used for fine-tuning. Optional.",
    "files.model_file": "Path to the trained model to be used. If training a new model, specify the desired output filename.",
    "files.config_file": "Path to the Detectron2 configuration file (.yaml). Required. Do not change unless using a custom config.",
    "files.test_metadata_catalog_file": "Path to the metadata catalog for testing. Optional, unless using a custom metadata structure.",
    "files.train_dataset_catalog_file": "Path to the training dataset catalog. Optional, unless using a custom dataset setup.",
    "files.train_annotation_file": "Path to the COCO-format annotation file for training. Required if 'download_assets' is disabled.",
    "files.test_annotation_file": "Path to the COCO-format annotation file for testing. Required if 'download_assets' is disabled.",
    "files.validation_annotation_file": "Path to the COCO-format annotation file for validation. Optional but recommended.",
    "files.model_url": "URL to download model weights. Required if 'download_assets' is enabled and no local model is provided.",
    "files.meta_catalog_url": "URL to download the metadata catalog. Required if 'download_assets' is enabled and no local catalog is provided.",
    "files.train_catalog_url": "URL to download the training dataset catalog. Required if 'download_assets' is enabled and no local file is provided.",

    # Directories
    "directories.train_image_dir": "Directory containing images for training. Required for both training and evaluation.",
    "directories.test_image_dir": "Directory containing images to be used for prediction. Required for inference mode.",
    "directories.validation_image_dir": "Directory containing validation images. Optional unless validation is enabled.",
    "directories.training_output_dir": "Directory where training logs and outputs will be saved.",
    "directories.prediction_output_dir": "Directory where predicted images will be saved. Required in both training and prediction modes.",
    "directories.prediction_json_dir": "Directory where prediction JSON files will be stored. Required in both training and prediction modes.",

    # Training parameters
    "training.epochs": "Number of training epochs to run. Must be a positive integer.",
    "training.batch_size": "Number of samples per batch during training. Typical values range from 2 to 32.",
    "training.number_of_classes": "Total number of classes the model should predict. Required.",
    "training.optimizer": "Optimizer algorithm to use for training (e.g., 'SGD', 'Adam').",
    "training.learning_rate": "Learning rate used by the optimizer. Typical values range from 0.001 to 0.01.",

    # Model settings
    "settings.download_assets": "Set to True to enable downloading of datasets and model files. Set to False if all assets are already available locally.",
    "settings.rename_pred_images": "Set to True to rename predicted image files using a numbered pattern (e.g., img_000001.png).",
    "settings.segm_masks": "Set to True to enable segmentation mask predictions. Set to False to disable.",
    "settings.bbox": "Set to True to enable bounding box predictions. Set to False to disable.",
    "settings.show_orientation": "Set to True to enable orientation estimation for detections. Only supported for strawberries and tomatoes.",
    "settings.fruit_type": "Name of the fruit to be detected. Supported values: 'strawberry', 'tomato'. Leave empty if using a model trained on a different fruit.",
    "settings.validation_period": "Applies during training. Smaller values increase validation frequency, which may increase training time.",
    "settings.confidence_threshold": "Removes predictions with confidence scores below this threshold.",
    "settings.filename_patterns.rgb": "Pattern used to identify RGB images by filename.",
    "settings.filename_patterns.depth": "Pattern used to identify depth images by filename."
}
