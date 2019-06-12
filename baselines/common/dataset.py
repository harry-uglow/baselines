from __future__ import absolute_import
import numpy as np
from itertools import imap

class Dataset(object):
    def __init__(self, data_map, deterministic=False, shuffle=True):
        self.data_map = data_map
        self.deterministic = deterministic
        self.enable_shuffle = shuffle
        self.n = iter(data_map.values()).next().shape[0]
        self._next_id = 0
        self.shuffle()

    def shuffle(self):
        if self.deterministic:
            return
        perm = np.arange(self.n)
        np.random.shuffle(perm)

        for key in self.data_map:
            self.data_map[key] = self.data_map[key][perm]

        self._next_id = 0

    def next_batch(self, batch_size):
        if self._next_id >= self.n and self.enable_shuffle:
            self.shuffle()

        cur_id = self._next_id
        cur_batch_size = min(batch_size, self.n - self._next_id)
        self._next_id += cur_batch_size

        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][cur_id:cur_id+cur_batch_size]
        return data_map

    def iterate_once(self, batch_size):
        if self.enable_shuffle: self.shuffle()

        while self._next_id <= self.n - batch_size:
            yield self.next_batch(batch_size)
        self._next_id = 0

    def subset(self, num_elements, deterministic=True):
        data_map = dict()
        for key in self.data_map:
            data_map[key] = self.data_map[key][:num_elements]
        return Dataset(data_map, deterministic)


def iterbatches(arrays, **_3to2kwargs):
    if 'include_final_partial_batch' in _3to2kwargs: include_final_partial_batch = _3to2kwargs['include_final_partial_batch']; del _3to2kwargs['include_final_partial_batch']
    else: include_final_partial_batch = True
    if 'shuffle' in _3to2kwargs: shuffle = _3to2kwargs['shuffle']; del _3to2kwargs['shuffle']
    else: shuffle = True
    if 'batch_size' in _3to2kwargs: batch_size = _3to2kwargs['batch_size']; del _3to2kwargs['batch_size']
    else: batch_size = None
    if 'num_batches' in _3to2kwargs: num_batches = _3to2kwargs['num_batches']; del _3to2kwargs['num_batches']
    else: num_batches = None
    assert (num_batches is None) != (batch_size is None), u'Provide num_batches or batch_size, but not both'
    arrays = tuple(imap(np.asarray, arrays))
    n = arrays[0].shape[0]
    assert all(a.shape[0] == n for a in arrays[1:])
    inds = np.arange(n)
    if shuffle: np.random.shuffle(inds)
    sections = np.arange(0, n, batch_size)[1:] if num_batches is None else num_batches
    for batch_inds in np.array_split(inds, sections):
        if include_final_partial_batch or len(batch_inds) == batch_size:
            yield tuple(a[batch_inds] for a in arrays)
