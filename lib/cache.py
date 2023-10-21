import torch
from lib.neighbor_sampler import *
from lib.utils import *
import threading
from queue import Queue
import numpy as np
import json
import os
from tqdm import tqdm


class FeatureCache:
    def __init__(self, size, num_nodes, mmapped_features, feature_dim):
        
        self.size = size
        self.num_nodes = num_nodes
        self.mmapped_features = mmapped_features
        self.feature_dim = feature_dim

        # The address table of the cache has num_nodes entries each of which is a single
        # int32 value. This can support the cache with up to 2147483647 entries.
        table_size = 4 * self.num_nodes
        self.num_entries = int((self.size-table_size)/4/self.feature_dim)
        if self.num_entries > torch.iinfo(torch.int32).max:
            raise ValueError


    def init_cache(self, indices):
        self.address_table = torch.full((self.num_nodes,), -2, dtype=torch.int32) # init value
        self.true_address_table = torch.full((self.num_nodes,), -2, dtype=torch.int32) # init value
        self.address_table[indices] = torch.full((indices.numel(),), -1, dtype=torch.int32) # -1 means waiting to be filled
        self.true_address_table[indices] = torch.arange(indices.numel(), dtype=torch.int32)

    def update_cache(self, n_id, batch_inputs):
        update_feat_cache(self, n_id, batch_inputs)


class NeighborCache:
    '''
    Neighbor cache

    Args:
        size (int): the size of the neighbor cache including both the cached data and 
            the address table in byte.
        score (Tensor): the score of each node defined as the ratio between the number 
            of out-neighbors and in-neighbors.
        indptr (Tensor): the indptr tensor.
        indices (Tensor): the (memory-mapped) indices tensor.
        num_nodes (int): the number of nodes in the graph.
    '''
    def __init__(self, size, score, indptr, indices, num_nodes):
        self.size = size
        self.indptr = indptr
        self.indices = indices
        self.num_nodes = num_nodes

        self.cache, self.address_table, self.num_entries = self.init_by_score(score)


    def init_by_score(self, score):
        sorted_indices = score.argsort(descending=True)
        # (1 - n) - (0 - n-1)
        neighbor_counts = self.indptr[1:] - self.indptr[:-1]
        neighbor_counts = neighbor_counts[sorted_indices]

        table_size = self.num_nodes*8 # int64
        # spare the space for pointers of all the vertices
        cache_size = int((self.size - table_size)/8)
        if cache_size < 0:
            raise ValueError

        address_table = torch.full((self.num_nodes,), -1, dtype=torch.int64)

        # Fetch neighborhood information of nodes into the cache one by one in order
        # of score until the cache gets full
        cumulative_size = torch.cumsum(neighbor_counts+1, dim=0)
        num_entries = (cumulative_size <= cache_size).sum().item()
        print(num_entries)
        # fill in the address_table 1, cumu[0], ..., cumu[n-1]
        address_table[sorted_indices[:num_entries]] = torch.cat([torch.zeros(1).long(), cumulative_size[:num_entries-1]])
        # shape: (cached_node_num, )
        cached_idx = (address_table >= 0).nonzero().squeeze()

        # Multi-threaded load of neighborhood information
        cache = torch.zeros(cache_size, dtype=torch.int64)
        fill_neighbor_cache(cache, self.indptr, self.indices, cached_idx, address_table, num_entries)
                    
        return cache, address_table, num_entries


    def save(self, data, filename):
        data_path = filename + '.dat'
        conf_path = filename + '_conf.json'

        data_mmap = np.memmap(data_path, mode='w+', shape=data.shape, dtype=data.dtype)
        data_mmap[:] = data[:]
        data_mmap.flush()

        mmap_config = dict()
        mmap_config['shape'] = tuple(data.shape)
        mmap_config['dtype'] = str(data.dtype)

        json.dump(mmap_config, open(conf_path, 'w'))
