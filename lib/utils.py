from lib.cpp_extension.wrapper import *


def tensor_free(t):
    free.tensor_free(t)


def tensor_ptr_free(ptr):
    free.tensor_ptr_free(ptr)


def get_tensor_ptr(t):
    return free.get_tensor_ptr(t)


def gather_normal(feature_file, idx, feature_dim, cache):
    return gather.gather_normal(feature_file, idx, feature_dim, cache.cache, cache.address_table)


def gather_zerocopy(features, idx, feature_dim, cache):
    return gather.gather_zerocopy(features, idx, feature_dim, cache.cache, cache.address_table)


def update_feat_cache(cache, n_id, batch_inputs):
    update.update_feat_cache(n_id, batch_inputs, cache.cache, cache.address_table, cache.true_address_table, cache.feature_dim)


def update_feat_cache_zerocopy(cache, n_id, batch_inputs):
    update.update_feat_cache_zerocopy(n_id, batch_inputs, cache.cache, cache.address_table, cache.true_address_table, cache.feature_dim)


def gather_direct(feature_file, idx, feature_dim):
    return gather.gather_direct(feature_file, idx, feature_dim)


def gather_mmap(features, idx):
    return gather.gather_mmap(features, idx, features.shape[1])


def load_float32(path, size1, size2):
    return mt_load.load_float32(path, size1, size2)


def load_int64(path, size):
    return mt_load.load_int64(path, size)


def cache_check(n_id, emb_cache_table, true_emb_cache_table, emb_cache_stale, n_iter):
    return update.cache_check(n_id, emb_cache_table, true_emb_cache_table, emb_cache_stale, n_iter)


def fill_neighbor_cache(cache, rowptr, col, cached_idx, address_table, num_entries):
    sample.fill_neighbor_cache(cache, rowptr, col, cached_idx, address_table, num_entries)


def fennel_partition(result, cross_edge_num, csr_indptr, csr_indices, num_partitions, num_rounds, num_train_nodes, train_mask, imbalance_ratio, gamma, patience, mapping):
    partition.fennel_partition(result, cross_edge_num, csr_indptr, csr_indices, num_partitions, num_rounds, num_train_nodes, train_mask, patience, imbalance_ratio, gamma, mapping)

def fennel_bf_partition(result, cross_edge_num, csr_indptr, csr_indices, num_partitions, num_rounds, num_train_nodes, train_mask, imbalance_ratio, gamma, patience, mapping):
    partition.fennel_bf_partition(result, cross_edge_num, csr_indptr, csr_indices, num_partitions, num_rounds, num_train_nodes, train_mask, patience, imbalance_ratio, gamma, mapping)


def fennel_partition_outofcore(result, cross_edge_num, csr_indptr, csr_indices_file, num_edges, num_partitions, num_rounds, num_train_nodes, train_mask, imbalance_ratio, gamma, patience, mapping):
    partition.fennel_partition_outofcore(result, cross_edge_num, csr_indptr, csr_indices_file, num_edges, num_partitions, num_rounds, num_train_nodes, train_mask, patience, imbalance_ratio, gamma, mapping)


def fennel_bf_partition_outofcore(result, cross_edge_num, csr_indptr, csr_indices_file, num_edges, num_partitions, num_rounds, num_train_nodes, train_mask, imbalance_ratio, gamma, patience, mapping):
    partition.fennel_bf_partition_outofcore(result, cross_edge_num, csr_indptr, csr_indices_file, num_edges, num_partitions, num_rounds, num_train_nodes, train_mask, patience, imbalance_ratio, gamma, mapping)


def fetch_split_nodes(inpart_nodes, mask):
    return partition.fetch_split_nodes(inpart_nodes, mask)


def fetch_csr(csr_indptr, csr_indices, inpart_nodes):
    return partition.fetch_csr(csr_indptr, csr_indices, inpart_nodes)


def fetch_csr_first_level(csr_indptr, csr_indices, inpart_nodes):
    return partition.fetch_csr_first_level(csr_indptr, csr_indices, inpart_nodes)


def fetch_csr_outofcore(csr_indptr, csr_indices_file, inpart_nodes):
    return partition.fetch_csr_outofcore(csr_indptr, csr_indices_file, inpart_nodes)


def fetch_csr_first_level_outofcore(csr_indptr, csr_indices_file, inpart_nodes):
    return partition.fetch_csr_first_level_outofcore(csr_indptr, csr_indices_file, inpart_nodes)
