from coinstac_sparse_dinunet.site_runner import SiteRunner
import sys
sys.path.append("/data/users2/bthapaliya/coinstac_dist_pruning/coinstac_cifar10")
from comps.cifar10 import CIFAR10Trainer, CIFAR10DataHandle, CIFAR10Dataset

if __name__ == "__main__":
    runner = SiteRunner(taks_id='CIFAR10-Classification', data_path='/data/users2/bthapaliya/coinstac_dist_pruning/coinstac_cifar10/test/input_CIFAR10', mode='Train', split_ratio=[0.8, 0.1, 0.1], gpus=0, pretrain_args=None)
    runner.run(CIFAR10Trainer, CIFAR10Dataset, CIFAR10DataHandle)
