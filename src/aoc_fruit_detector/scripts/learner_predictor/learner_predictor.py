"""
Started by: Usman Zahidi (uz) {20/08/24}

"""
# general imports
import abc

# abstract class for DetectronPredictor, in future, for example YOLOPredictor will be extended similarly.

class LearnerPredictor(abc.ABC):

    @abc.abstractmethod
    def _configure(self):
        pass

    @abc.abstractmethod
    def get_predictions(self):
        pass
