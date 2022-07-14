import multiprocessing as mp
import time

from coinstac_sparse_dinunet import COINNLocal
from coinstac_sparse_dinunet.utils import duration

from comps import AggEngine
from comps import NNComputation, FreeSurferDataset, FreeSurferTrainer, FSVDataHandle
from comps import ICATrainer, ICADataset, ICADataHandle

""" Test """
computation = NNComputation.TASK_FREE_SURFER
agg_engine = AggEngine.DECENTRALIZED_SGD

CACHE = {}
MP_POOL = None


def run(data):
    global CACHE
    global MP_POOL

    _start = time.time()
    start_time = CACHE.setdefault('start_time', _start)

    if MP_POOL is None and CACHE.get('num_reducers'):
        MP_POOL = mp.Pool(processes=CACHE['num_reducers'])

    dataloader_args = {"train": {"drop_last": True}}

    local = COINNLocal(
        task_id=computation, agg_engine=agg_engine,
        cache=CACHE, input=data['input'], batch_size=16,
        state=data['state'], epochs=21, patience=31, split_ratio=[0.7, 0.15, 0.15],
        pretrain_args=None, dataloader_args=dataloader_args,
        num_class=2, monitor_metric='auc', log_header="loss|auc"
    )

    """Add new NN computation Here"""
    if local.cache['task_id'] == NNComputation.TASK_FREE_SURFER:
        args = FreeSurferTrainer, FreeSurferDataset, FSVDataHandle

    elif local.cache['task_id'] == NNComputation.TASK_ICA:
        args = ICATrainer, ICADataset, ICADataHandle

    else:
        raise ValueError(f"Invalid local task:{local.cache.get('task')}")

    out = local(MP_POOL, *args)

    duration(CACHE, _start, key='time_spent_on_computation')
    duration(CACHE, start_time, key='cumulative_total_duration')
    return out
