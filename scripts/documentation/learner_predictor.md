# `learner_predictor.py`

## Purpose

Defines `LearnerPredictor`, an abstract base class specifying the interface expected of prediction implementations. The source comments identify `DetectronPredictor` as an implementation and suggest that a future `YOLOPredictor` could follow the same interface.

## Dependencies

- Python standard-library `abc` module, used for abstract base classes and abstract methods.

## Class: `LearnerPredictor`

```python
class LearnerPredictor(abc.ABC):
```

The class extends `abc.ABC`, so subclasses must implement its abstract methods before they can be instantiated.

| Abstract method | Intended responsibility | Arguments declared in this base class |
|---|---|---|
| `_configure(self)` | Configure the prediction implementation | None beyond `self` |
| `get_predictions_message(self)` | Produce predictions in message form | None beyond `self` |
| `get_predictions_image(self)` | Produce predictions for an image | None beyond `self` |

Each method is decorated with `@abc.abstractmethod` and contains only `pass`. The file does not provide inference logic, model loading, data preparation, or output formatting.

## Relationship to `DetectronPredictor`

The source explicitly describes this as the abstract class for `DetectronPredictor`. A concrete predictor must provide implementations of the three declared abstract methods. The interface can also be extended for other predictor backends, such as the proposed YOLO implementation.

## Usage

This class is not intended to be instantiated directly. A minimal implementation would look like:

```python
from learner_predictor.learner_predictor import LearnerPredictor

class ExamplePredictor(LearnerPredictor):
    def _configure(self):
        return None

    def get_predictions_message(self):
        return None

    def get_predictions_image(self):
        return None
```

This example only illustrates satisfaction of the abstract interface; it does not implement prediction.

## Implementation notes

- The abstract methods do **not** define concrete argument lists or return types beyond `self`; subclasses may expose additional parameters.
- There is no shared implementation, validation, error handling, or configuration storage in the base class.
- Python's abstract-method enforcement checks that concrete methods exist, but does not enforce identical method signatures or outputs.
