# `learner_trainer.py` — LearnerTrainer Interface

## Purpose

Defines `LearnerTrainer`, an abstract base class for model-training implementations. It establishes a minimal common interface intended for trainers such as `DetectronTrainer` and potentially other frameworks (the source mentions a future `YOLOTrainer`).

## Dependencies

- Python standard library `abc` module, used to define abstract methods.

## Class: `LearnerTrainer`

```python
class LearnerTrainer(abc.ABC):
```

Inheriting from `abc.ABC` makes this an abstract base class. Subclasses must implement the following methods before they can be instantiated:

| Method | Purpose | Defined behaviour |
|---|---|---|
| `_configure(self)` | Provide a trainer configuration interface | Abstract placeholder (`pass`) |
| `train_model(self)` | Provide a model training interface | Abstract placeholder (`pass`) |

Both methods are decorated with `@abc.abstractmethod`. The base class does **not** implement configuration or training logic.

## Expected inheritance

A concrete implementation subclasses `LearnerTrainer` and implements both methods. For example, the project’s `DetectronTrainer` subclasses this interface and provides its own configuration and training workflow.

```python
from learner_trainer.learner_trainer import LearnerTrainer

class ExampleTrainer(LearnerTrainer):
    def _configure(self):
        return {"configured": True}

    def train_model(self):
        return "training completed"
```

This example illustrates the required interface; it is not part of the uploaded script.

## Behaviour and limitations

- There is no constructor, model definition, optimizer, dataset loader, evaluation routine, or training loop in this file.
- The base class specifies no arguments beyond `self` for either abstract method. Concrete implementations may extend their signatures.
- The methods include no type annotations, return-value requirements, or error-handling contracts.
- Instantiating `LearnerTrainer` directly raises a `TypeError` because its abstract methods are unimplemented.

## Source

Documented from `learner_trainer.py` (header credits Usman Zahidi, 20/08/24).
