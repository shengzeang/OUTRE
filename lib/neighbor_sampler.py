from typing import List, Optional, Tuple, NamedTuple, Callable
import os
import time
import gc
import numpy as np
import json
import torch
from torch import Tensor
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import spmm
import torch.multiprocessing as mp
import multiprocessing
from queue import Queue
import threading

from lib.cpp_extension.wrapper import sample
from lib.data import *
from lib.cache import *
from lib.utils import *


class Adj(NamedTuple):
    adj_t: SparseTensor
    e_id: Optional[Tensor]
    size: Tuple[int, int]


    def to(self, *args, **kwargs):
        adj_t = self.adj_t.to(*args, **kwargs)
        e_id = self.e_id.to(*args, **kwargs) if self.e_id is not None else None
        return Adj(adj_t, e_id, self.size)


class PartitionNeighborSampler(torch.utils.data.DataLoader):
    '''
    Neighbor sampler of OUTRE. We modified NeighborSampler class of Ginex.

    Args:
        indptr (Tensor): the indptr tensor.
        indices (Tensor): the (memory-mapped) indices tensor.
        exp_name (str): the name of the experiments used to designate the path of the
            runtime trace files.
        sb (int): the superbatch number.
        sizes ([int]): The number of neighbors to sample for each node in each layer.
            If set to sizes[l] = -1`, all neighbors are included in layer `l`.
        node_idx (Tensor): The nodes that should be considered for creating mini-batches.
        cache_data (Tensor): the data array of the neighbor cache.
        address_table (Tensor): the address table of the neighbor cache.
        num_nodes (int): the number of nodes in the graph.
        transform (callable, optional): A function/transform that takes in a sampled
            mini-batch and returns a transformed version. (default: None)
        **kwargs (optional): Additional arguments of
            `torch.utils.data.DataLoader`, such as `batch_size`,
            `shuffle`, `drop_last`m `num_workers`.
    '''
    def __init__(self, indptr, indices, exp_name, partition_num, partition_batch, partition_path,
                 sizes: List[int], emb_cache_table=None, mode=None,
                 true_emb_cache_table=None, emb_cache_stale=None,
                 cache_data = None, address_table = None, batch_size=None,
                 num_nodes: Optional[int] = None,
                 transform: Callable = None, **kwargs):

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        self.indptr = indptr
        self.indices = indices
        self.exp_name = exp_name
        self.num_nodes = num_nodes

        self.batch_size_global = batch_size

        self.partition_num = partition_num
        self.partition_batch = partition_batch
        self.partition_path = partition_path
        self.mode = mode

        if emb_cache_table is None:
            self.emb_cache_table = None
            self.emb_cache_stale = None
            self.true_emb_cache_table = None
        else:
            self.emb_cache_table = emb_cache_table.share_memory_()
            self.emb_cache_stale = emb_cache_stale.share_memory_()
            self.true_emb_cache_table = true_emb_cache_table.share_memory_()

        self.access_freq = torch.zeros(num_nodes).share_memory_()
        self.neigh_size_before = torch.zeros(num_nodes).share_memory_()
        self.neigh_size_after = torch.zeros(num_nodes).share_memory_()

        self.cache_data = cache_data
        self.address_table = address_table

        self.sizes = sizes
        self.transform = transform

        self.batch_count = torch.zeros(1, dtype=torch.int).share_memory_()
        self.lock = mp.Lock()

        super(PartitionNeighborSampler, self).__init__(
            torch.randperm(self.partition_num).numpy().tolist(), batch_size=self.partition_batch, collate_fn=self.sample, **kwargs)


    def load_csr_by_batch_full(self, batch):
        # print(batch)
        if self.mode == 'train' or self.mode == 'profiling':
            chosen_n_id = torch.load(self.partition_path+f"/part{batch[0]}/train_n_id.dat")
        elif self.mode == 'valid':
            chosen_n_id = torch.load(self.partition_path+f"/part{batch[0]}/val_n_id.dat")
        elif self.mode == 'test':
            chosen_n_id = torch.load(self.partition_path+f"/part{batch[0]}/test_n_id.dat")

        n_id = torch.load(self.partition_path+f"/part{batch[0]}/n_id.dat")
        csr_indptr = torch.load(self.partition_path+f"/part{batch[0]}/csr_indptr.dat")
        csr_indices = torch.load(self.partition_path+f"/part{batch[0]}/csr_indices.dat")

        for id in batch[1:]:
            if self.mode == 'train' or self.mode == 'profiling':
                chosen_n_id = torch.hstack((chosen_n_id, torch.load(self.partition_path+f"/part{id}/train_n_id.dat")))
            elif self.mode == 'valid':
                chosen_n_id = torch.hstack((chosen_n_id, torch.load(self.partition_path+f"/part{id}/val_n_id.dat")))
            elif self.mode == 'test':
                chosen_n_id = torch.hstack((chosen_n_id, torch.load(self.partition_path+f"/part{id}/test_n_id.dat")))

            n_id = torch.hstack((n_id, torch.load(self.partition_path+f"/part{id}/n_id.dat")))
            csr_indptr = torch.hstack((csr_indptr[:-1], csr_indptr[-1]+torch.load(self.partition_path+f"/part{id}/csr_indptr.dat")))
            csr_indices = torch.hstack((csr_indices, torch.load(self.partition_path+f"/part{id}/csr_indices.dat")))

        return chosen_n_id, n_id, csr_indptr.long(), csr_indices.long()


    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)

        chosen_n_id, n_id, csr_indptr, csr_indices = self.load_csr_by_batch_full(batch)

        partition_size = len(chosen_n_id)
        shuffled_n_id_idx = torch.randperm(len(chosen_n_id))
        chosen_n_id = chosen_n_id[shuffled_n_id_idx]

        if partition_size > self.batch_size_global:
            batch_num = int(partition_size / self.batch_size_global)
            if partition_size % self.batch_size_global != 0:
                batch_num += 1

            for st in range(0, batch_num):
                adjs = []
                chosen_n_id_batch = chosen_n_id[st*self.batch_size_global : min((st+1)*self.batch_size_global, partition_size)]

                self.lock.acquire()
                batch_count = self.batch_count.item()
                n_id_filename = os.path.join('./trace', self.exp_name, '_ids_' + str(batch_count) + '.pth')
                adjs_filename = os.path.join('./trace', self.exp_name, '_adjs_' + str(batch_count) + '.pth')
                self.batch_count += 1
                self.lock.release()

                if self.mode == 'profiling':
                    adjs_neigh_batch = []
                adjs_batch = []
                start_idx = 0
                pre_row_ptr, pre_cols = None, None
                one_hop_n_id_batch = None
                for (i, size) in enumerate(self.sizes):
                    if self.emb_cache_table != None and i >= 1:
                        rowptr, col, new_chosen_n_id_batch = sample.sample_adj_prune_rm_repeat(
                            self.indptr, self.indices, chosen_n_id_batch, pre_row_ptr, pre_cols, start_idx, self.cache_data, self.address_table, n_id, csr_indptr, csr_indices, self.emb_cache_table, self.true_emb_cache_table, self.emb_cache_stale, batch_count, i, size, False)
                    else:
                        if self.mode == 'profiling':
                            rowptr, col, new_chosen_n_id_batch, rowptr_neigh, col_neigh = sample.sample_adj_profiling(
                                self.indptr, self.indices, chosen_n_id_batch, self.cache_data, self.address_table, n_id, csr_indptr, csr_indices, size, False)
                        else:
                            rowptr, col, new_chosen_n_id_batch = sample.sample_adj(
                                self.indptr, self.indices, chosen_n_id_batch, self.cache_data, self.address_table, n_id, csr_indptr, csr_indices, size, False)

                    if self.mode == 'profiling':
                        adj_t_neigh_batch = SparseTensor(rowptr=rowptr_neigh, row=None, col=col_neigh,
                            sparse_sizes=(chosen_n_id_batch.size(0), new_chosen_n_id_batch.size(0)),
                            is_sorted=True)
                    adj_t_batch = SparseTensor(rowptr=rowptr, row=None, col=col,
                            sparse_sizes=(chosen_n_id_batch.size(0), new_chosen_n_id_batch.size(0)),
                            is_sorted=True)
                    start_idx = chosen_n_id_batch.size(0)
                    pre_row_ptr = rowptr
                    pre_cols = col
                    chosen_n_id_batch = new_chosen_n_id_batch
                    if i == 0:
                        one_hop_n_id_batch = new_chosen_n_id_batch

                    if self.mode == 'profiling':
                        e_id = adj_t_neigh_batch.storage.value()
                        size = adj_t_neigh_batch.sparse_sizes()[::-1]
                        adjs_neigh_batch.append(Adj(adj_t_neigh_batch, e_id, size))

                    e_id = adj_t_batch.storage.value()
                    size = adj_t_batch.sparse_sizes()[::-1]
                    adjs_batch.append(Adj(adj_t_batch, e_id, size))

                if self.mode == 'profiling':
                    adjs_neigh_batch = adjs_neigh_batch if len(adjs_neigh_batch) == 1 else adjs_neigh_batch[::-1]
                adjs_batch = adjs_batch if len(adjs_batch) == 1 else adjs_batch[::-1]

                if self.mode == 'profiling':
                    neigh_size_before = torch.ones((len(chosen_n_id_batch), 1))
                    for i, (adj_batch, _, _) in enumerate(adjs_batch[:-1]):
                        neigh_size_before = spmm(adj_batch, neigh_size_before)
                        if i != len(adjs_batch) - 2:
                            neigh_size_before += torch.ones((adj_batch.size(0), 1))

                    neigh_size_after = torch.ones((len(chosen_n_id_batch), 1))
                    for i, (adj_neigh_batch, _, _) in enumerate(adjs_neigh_batch[:-1]):
                        neigh_size_after = spmm(adj_neigh_batch, neigh_size_after)
                        if i != len(adjs_neigh_batch) - 2:
                            neigh_size_after += torch.ones((adj_neigh_batch.size(0), 1))

                self.lock.acquire()
                if self.mode == 'profiling':
                    self.access_freq[chosen_n_id_batch] += 1
                    self.neigh_size_before[one_hop_n_id_batch] += neigh_size_before.squeeze(1)
                    self.neigh_size_after[one_hop_n_id_batch] += neigh_size_after.squeeze(1)
                self.lock.release()
                torch.save(chosen_n_id_batch, n_id_filename)
                torch.save(adjs_batch, adjs_filename)

        else:
            self.lock.acquire()
            batch_count = self.batch_count.item()
            n_id_filename = os.path.join('./trace', self.exp_name, '_ids_' + str(batch_count) + '.pth')
            adjs_filename = os.path.join('./trace', self.exp_name, '_adjs_' + str(batch_count) + '.pth')
            self.batch_count += 1
            self.lock.release()

            if self.mode == 'profiling':
                adjs_neigh = []
            adjs = []
            start_idx = 0
            pre_row_ptr, pre_cols = None, None
            one_hop_n_id = None
            for (i, size) in enumerate(self.sizes):
                if self.emb_cache_table != None and i >= 1:
                    rowptr, col, new_chosen_n_id = sample.sample_adj_prune_rm_repeat(
                        self.indptr, self.indices, chosen_n_id, pre_row_ptr, pre_cols, start_idx, self.cache_data, self.address_table, n_id, csr_indptr, csr_indices, self.emb_cache_table, self.true_emb_cache_table, self.emb_cache_stale, batch_count, i, size, False)
                else:
                    if self.mode == 'profiling':
                        rowptr, col, new_chosen_n_id, rowptr_neigh, col_neigh = sample.sample_adj_profiling(
                            self.indptr, self.indices, chosen_n_id, self.cache_data, self.address_table, n_id, csr_indptr, csr_indices, size, False)
                    else:
                        rowptr, col, new_chosen_n_id = sample.sample_adj(
                            self.indptr, self.indices, chosen_n_id, self.cache_data, self.address_table, n_id, csr_indptr, csr_indices, size, False)

                if self.mode == 'profiling':
                    adj_t_neigh = SparseTensor(rowptr=rowptr_neigh, row=None, col=col_neigh,
                        sparse_sizes=(chosen_n_id.size(0), new_chosen_n_id.size(0)),
                        is_sorted=True)
                adj_t = SparseTensor(rowptr=rowptr, row=None, col=col,
                        sparse_sizes=(chosen_n_id.size(0), new_chosen_n_id.size(0)),
                        is_sorted=True)
                start_idx = chosen_n_id.size(0)
                pre_row_ptr = rowptr
                pre_cols = col
                chosen_n_id = new_chosen_n_id
                if i == 0:
                    one_hop_n_id = new_chosen_n_id

                if self.mode == 'profiling':
                    e_id = adj_t_neigh.storage.value()
                    size = adj_t_neigh.sparse_sizes()[::-1]
                    adjs_neigh.append(Adj(adj_t_neigh, e_id, size))

                e_id = adj_t.storage.value()
                size = adj_t.sparse_sizes()[::-1]
                adjs.append(Adj(adj_t, e_id, size))

            if self.mode == 'profiling':
                adjs_neigh = adjs_neigh if len(adjs_neigh) == 1 else adjs_neigh[::-1]
            adjs = adjs if len(adjs) == 1 else adjs[::-1]

            if self.mode == 'profiling':
                neigh_size_before = torch.ones((len(chosen_n_id), 1))
                for i, (adj, _, _) in enumerate(adjs[:-1]):
                    neigh_size_before = spmm(adj, neigh_size_before)
                    if i != len(adjs) - 2:
                        neigh_size_before += torch.ones((adj.size(0), 1))

                neigh_size_after = torch.ones((len(chosen_n_id), 1))
                for i, (adj_neigh, _, _) in enumerate(adjs_neigh[:-1]):
                    neigh_size_after = spmm(adj_neigh, neigh_size_after)
                    if i != len(adjs_neigh) - 2:
                        neigh_size_after += torch.ones((adj_neigh.size(0), 1))

            self.lock.acquire()
            if self.mode == 'profiling':
                self.access_freq[chosen_n_id] += 1
                self.neigh_size_before[one_hop_n_id] += neigh_size_before.squeeze(1)
                self.neigh_size_after[one_hop_n_id] += neigh_size_after.squeeze(1)
            self.lock.release()
            torch.save(chosen_n_id, n_id_filename)
            torch.save(adjs, adjs_filename)


    def __repr__(self):
        return '{}(sizes={})'.format(self.__class__.__name__, self.sizes)
