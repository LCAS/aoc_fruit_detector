"""
Started by: Usman Zahidi (uz) {20/08/24}

"""
# general imports
import abc

# abstract class for DetectronTrainer, in future, for example YOLOTrainer will be extended similarly.

class LearnerTrainer(abc.ABC):

    @abc.abstractmethod
    def _configure(self):
        pass

    @abc.abstractmethod
    def train_model(self):
        pass
