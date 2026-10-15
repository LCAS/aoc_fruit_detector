# `utils.py` — Download and Path Utilities

## Overview

`utils.py` provides shared utilities for the fruit detection project's trainer and predictor. It has two responsibilities:

1. `LearnerUtils`: download model weights, dataset annotations and image archives, and serialized Detectron2 catalogs.
2. `find_path()` and `find_workspace_root()`: locate files/directories and identify a workspace root from the current working directory.

The script defines helpers; it does **not** execute training or inference itself.

## Dependencies

- Python standard library: `os`, `logging`, `traceback`, `shutil`, `zipfile`.
- External package: `requests` (HTTP downloads).

## `LearnerUtils`

### Initialization

```python
utils = LearnerUtils(config_data)
```

The constructor expects a nested configuration dictionary with the following keys:

| Section | Required keys |
| --- | --- |
| `files` | `model_file`, `train_annotation_file`, `test_annotation_file`, `test_metadata_catalog_file`, `model_url`, `meta_catalog_url`, `train_catalog_url`, `train_dataset_catalog_file` |
| `datasets` | `train_dataset_name`, `test_dataset_name`, `dataset_train_annotation_url`, `dataset_train_images_url`, `dataset_test_annotation_url`, `dataset_test_images_url` |
| `directories` | `train_image_dir`, `test_image_dir` |

Missing keys raise `KeyError` during construction.

### `call_download() -> None`

Main entry point for obtaining assets:

1. If `model_file` does not exist, downloads the model from `model_url`.
2. Calls `_download_datasets()` separately for train and test assets.
3. Calls `_download_catalogs()` for the test metadata and train dataset catalogs.

```python
utils = LearnerUtils(config_data)
utils.call_download()
```

### `_download_datasets(train_annotation_file, dataset_train_annotation_url, train_dataset_name, train_image_dir, dataset_images_url)`

Despite the argument names, this helper is used for **both** training and test datasets.

- If the annotation file already exists, it does nothing.
- Otherwise it downloads the annotation file and `<dataset_name>.zip` into the image directory.
- It extracts the ZIP into the image directory using `zipfile.ZipFile.extractall()`.

**Important:** The annotation file is the only existence check. If annotations exist but images are missing or incomplete, no image download is attempted. The ZIP archive is not removed afterward.

### `_download_catalogs(test_metadata_catalog_file, meta_catalog_url, train_dataset_catalog_file, train_catalog_url)`

When the **test metadata catalog** is absent, downloads both it and the train dataset catalog. If the test metadata catalog exists, the method skips both downloads even if the train catalog is missing.

### `_download(url: str, folder_name: str, local_filename: str) -> None`

Creates the destination directory if necessary, streams an HTTP response using `requests.get(url, stream=True)`, and writes its raw contents to the requested filename. On an exception, logs the error, prints a traceback in normal Python debug mode, and raises an exception.

**Important:** The code does not call `raise_for_status()`, so HTTP error responses may be written as files without raising an HTTP-status exception. There is no integrity verification or retry mechanism.

## Filesystem helpers

### `find_path(name, search_root='.', search_type='any')`

Walks the directory tree beneath `search_root` and returns the first matching path, or `None`.

| `search_type` | Behaviour |
| --- | --- |
| `'file'` | Looks for a filename equal to `name` in each visited directory. |
| `'dir'` | Looks for a directory name equal to `name`. |
| `'any'` | Tests whether `os.path.join(root, name)` is a file or directory. |

The function normalizes `name` using `os.path.normpath()` and prints its search progress and result.

```python
config_dir = find_path('config', search_root='/workspace/src', search_type='dir')
config_file = find_path('non_ros_params.yaml', search_root=config_dir, search_type='file')
```

**Limitations:** The `'file'` and `'dir'` modes are intended for simple names, not arbitrary relative paths; `'any'` can also match relative paths. Results depend on filesystem traversal order, and the returned path is not guaranteed to be absolute.

### `find_workspace_root(root_folder_name='fruit_detector_ws')`

Inspects `os.getcwd()` and splits its path into components. It returns the path through the **first** component equal to `root_folder_name`.

```python
workspace_root = find_workspace_root('fruit_detector_ws')
```

If the directory name is not present in the current working directory path, it raises `FileNotFoundError`. It does not search the filesystem or inspect ROS workspace metadata.

## Typical integration

`LearnerUtils` is used by the trainer/predictor to obtain configured assets. The path helpers are used by the standalone prediction workflow to locate its workspace and YAML configuration before building a predictor.

## Implementation notes and risks

- **Partial datasets:** Dataset downloads are skipped when the annotation JSON exists, regardless of image completeness.
- **Catalog dependency:** Both catalog downloads are gated on the test metadata file's existence.
- **HTTP validation:** No explicit status-code check, checksum, download timeout, or retry.
- **Archive extraction:** ZIP files are extracted without validating member paths; only use trusted archives.
- **Path lookup:** `find_workspace_root()` relies on the process's current directory containing the expected workspace name.
- **Error wrapping:** `_download()` catches exceptions and re-raises `Exception(e)`, which changes the original exception type.

These observations describe the uploaded implementation; they are not additional features supplied by the utility.
