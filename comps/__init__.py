from enum import Enum

from .fs import *
from .cifar10 import *



class NNComputation(str, Enum):
    """ Available tasks """
    TASK_FREE_SURFER = "FS-Classification"
    TASK_CIFAR10 = "CIFAR10-Classification"


class AggEngine(str, Enum):
    DECENTRALIZED_SGD = "dSGD"

