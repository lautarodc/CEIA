import numpy as np


class IndexerVector(object):
    def __init__(self, users):
        unique_users, idx = np.unique(users, return_index=True)
        self.idx2id = users[np.sort(idx)]
        indices = np.arange(0, np.size(idx), 1)
        self.id2idx = np.ones(np.max(users) + 1, dtype=np.int64) * -1
        self.id2idx[self.idx2id] = indices

    def get_idx(self, ids):
        return self.id2idx[ids], self.id2idx[ids] != -1

    def get_id(self, idx):
        return self.idx2id[idx[idx > 0]]
