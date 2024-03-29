import multiprocessing as mp
import time

from coinstac_sparse_dinunet import COINNRemote
from coinstac_sparse_dinunet.utils import duration

from comps import NNComputation, FreeSurferTrainer
from comps import NNComputation, CIFAR10Dataset, CIFAR10Trainer, CIFAR10DataHandle

CACHE = {}
MP_POOL = None


def run(data):
    global CACHE
    global MP_POOL

    _start = time.time()
    start_time = CACHE.setdefault('start_time', _start)

    if MP_POOL is None and CACHE.get('num_reducers'):
        MP_POOL = mp.Pool(processes=CACHE['num_reducers'])

    remote = COINNRemote(
        cache=CACHE, input=data['input'], state=data['state']
    )
    remote.cache['task_id'] = NNComputation.TASK_CIFAR10

    """Add new NN computation Here"""
    if remote.cache['task_id'] == NNComputation.TASK_FREE_SURFER:
        args = FreeSurferTrainer,
    elif remote.cache['task_id'] == NNComputation.TASK_CIFAR10:
        args = CIFAR10Trainer,
    else:
        raise ValueError(f"Invalid remote task:{remote.cache.get('task')}")

    out = remote(MP_POOL, *args)

    duration(CACHE, _start, key='time_spent_on_computation')
    duration(CACHE, start_time, key='cumulative_total_duration')
    return out
