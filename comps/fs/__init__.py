import os

import pandas as pd
import torch
import torch.nn.functional as F
from coinstac_sparse_dinunet import COINNDataset, COINNDataHandle
from coinstac_sparse_dinunet import COINNTrainer

from .models import MSANNet


class FreeSurferDataset(COINNDataset):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.labels = None

    def load_index(self, file):

        if self.labels is None:
            self.labels = pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache["labels_file"])
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
        df = pd.read_csv(self.path() + os.sep + file, sep='\t', names=['File', file], skiprows=1)
        df = df.set_index(df.columns[0])
        df = df / df.max().astype('float64')
        x = df.T.iloc[0].values
        return {'inputs': torch.tensor(x), 'labels': torch.tensor(y), 'ix': torch.tensor(ix)}


class FreeSurferTrainer(COINNTrainer):
    def __init__(self, **kw):
        super().__init__(**kw)

    def _init_nn_model(self):
        self.nn['fs_net'] = MSANNet(in_size=self.cache['input_size'],
                                    hidden_sizes=self.cache['hidden_sizes'], out_size=self.cache['num_class'])

    def iteration(self, batch):
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        out = F.log_softmax(self.nn['fs_net'](inputs), 1)
        loss = F.nll_loss(out, labels)

        _, predicted = torch.max(out, 1)
        score = self.new_metrics()
        score.add(predicted, labels)
        val = self.new_averages()
        val.add(loss.item(), len(inputs))
        return {'out': out, 'loss': loss, 'averages': val, 'metrics': score, 'prediction': predicted,
                'indices': indices}

    def single_iteration_for_masking(self, model, batch):
        sparsity_level = 0.85
        inputs, labels = batch['inputs'].to(self.device['gpu']).float(), batch['labels'].to(self.device['gpu']).long()
        indices = batch['ix'].to(self.device['gpu']).long()

        model.zero_grad()
        out = F.log_softmax(model.forward(inputs), 1)
        loss = F.nll_loss(out, labels)
        return {'out': out, 'loss': loss, 'indices': indices, 'sparsity_level': sparsity_level}


class FSVDataHandle(COINNDataHandle):
    def list_files(self):
        df = pd.read_csv(self.state['baseDirectory'] + os.sep + self.cache["labels_file"])
        if self.cache.get('data_column') in df.columns:
            df = df.set_index(self.cache['data_column'])
        return list(df.index)
