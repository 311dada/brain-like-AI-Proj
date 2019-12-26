import sys
sys.path.append("/mnt/lustre/sjtu/home/xnx98/utils/")
import kaldi_io

import itertools
import numpy as np
import torch
import torch.utils.data as data
import pdb

class Timit_dataset(data.Dataset):

    def __init__(self, feat_stream, label_stream, transform=None):
        self._features = {k: feats for k, feats in kaldi_io.read_mat_ark(feat_stream)}
        self._ali_label = {k: labels for k, labels in kaldi_io.read_ali_ark(label_stream)}
        self._transform = transform
    
    def __getitem__(self, idx):
        """
        idx: tuple, (kaldi_key, index)
        """
        kaldi_key, index = idx
        feat = self._features[kaldi_key][index]
        label = self._ali_label[kaldi_key][index]
        if self._transform:
            feat = self._transform(feat.reshape(1, -1)).reshape(-1)
        return feat, label

    def __len__(self):
        return sum(map(lambda item: len(item[1]), self.ali_label.items()))


class Timit_sampler(data.Sampler):

    def __init__(self, label_stream, shuffle):
        self.kaldikey2framenum = {k: len(labels) for k, labels in kaldi_io.read_ali_ark(label_stream)}
        # self.key2framenum_iter = iter(self.key2framenum)
        # self.cur_key = next(self.key2framenum_iter)
        # self.cur_idx = 0
        self.dataset_keys = []
        for kaldi_key, labels in kaldi_io.read_ali_ark(label_stream):
            self.dataset_keys.extend(itertools.product((kaldi_key,), range(len(labels))))
        if shuffle:
            np.random.shuffle(self.dataset_keys)

    def __iter__(self):
        return iter(self.dataset_keys)

    def __len__(self):
        return len(self.dataset_keys)

    # def __next__(self):
        # if self.cur_idx < self.key2framenum[self.cur_key]:
            # self.cur_idx += 1
            # return (self.cur_key, self.cur_idx - 1)
        # else:
            # try:
                # self.cur_key = next(self.key2framenum_iter)
                # self.cur_idx = 1
                # return (self.cur_key, self.cur_idx - 1)
            # except:
                # raise StopIteration


def create_dataloader(feat_stream,
                      label_stream,
                      transform=None,
                      batch_size=1024,
                      shuffle=True,
                      num_workers=16):
    dataset = Timit_dataset(feat_stream, label_stream, transform=transform)
    sampler = Timit_sampler(label_stream, shuffle)
    return data.DataLoader(dataset=dataset,
                           sampler=sampler,
                           batch_size=batch_size,
                           num_workers=num_workers)
    
