import os
import pandas as pd
import torch
import torchvision
import torch.nn.functional as F
from coinstac_sparse_dinunet import COINNDataset, COINNDataHandle
from coinstac_sparse_dinunet import COINNTrainer
import sys

print(sys.path)
from .resnet_model import resnet20, resnet32, resnet44, resnet56, resnet110, resnet1202

cifar10Labels = {
    "airplane": 0,
    "automobile": 1,
    "bird": 2,
    "cat": 3,
    "deer": 4,
    "dog": 5,
    "frog": 6,
    "horse": 7,
    "ship": 8,
    "truck": 9
}


class CIFAR10Dataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None

    def load_index(self, file):

        if self.labels is None:
            self.labels = pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache["covariates"])
            if self.cache.get('data_column') in self.labels.columns:
                self.labels = self.labels.set_index(self.cache['data_column'])

        y = self.labels.loc[file][self.cache['labels_column']]

        if isinstance(y, str):
            y = int(y.strip().lower() == 'true')

        """
        int64 could not be json serializable.
        """
        self.indices.append([file, int(y)])

    def __getitem__(self, ix):
        file, y = self.indices[ix]
        y = self.labels.iloc[ix, :]["label"]
        y = cifar10Labels.get(y)
        x = torchvision.io.read_image(self.path() + self.labels.iloc[ix, :]["cifar10_file_name"])
        # df = pd.read_csv(self.path() + os.sep + file, sep='\t', names=['File', file], skiprows=1)
        # df = df.set_index(df.columns[0])
        # df = df / df.max().astype('float64')
        # x = df.T.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


class CIFAR10Trainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        resnetmodel = globals()[self.cache["resnetmodel"]]
        self.nn['cifar10_resnet'] = resnetmodel()
        # self.nn['cifar10_resnet'] = resnet
        # self.nn['fs_net'] = MSANNet(in_size=self.cache['input_size'],
        #                             hidden_sizes=self.cache['hidden_sizes'], out_size=self.cache['num_class'])

    def single_iteration_for_masking(self, model, batch):
        sparsity_level = 0.90
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()
        model.zero_grad()
        out = F.log_softmax(model.forward(inputs), 1)
        loss = F.nll_loss(out, labels)
        return {'out': out, 'loss': loss, 'indices': indices, 'sparsity_level': sparsity_level}

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['cifar10_resnet'](inputs), 1)
        loss = F.nll_loss(out, labels)

        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': predicted,
                'indices': indices}


class CIFAR10DataHandle(COINNDataHandle):
    def list_files(self):
        df = pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache["covariates"])
        if self.cache.get('data_column') in df.columns:
            df = df.set_index(self.cache['data_column'])
        return list(df.index)
