import argparse
from ogb.nodeproppred import PygNodePropPredDataset
import scipy
import numpy as np
import json
import torch
import os
import networkx as nx

from torch_geometric.utils import to_undirected


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='ogbn-papers100m')
args = argparser.parse_args()

# Download/load dataset
print('Loading dataset...')
root = './dataset'
os.makedirs(root, exist_ok=True)
dataset = PygNodePropPredDataset(args.dataset, root)
dataset_path = os.path.join(root, args.dataset + '-new')
print('Done!')

# Construct sparse formats
print('Creating coo/csc/csr format of dataset...')
num_nodes = dataset[0].num_nodes
coo = to_undirected(dataset[0].edge_index).numpy()
v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
csc = coo.tocsc()
csr = coo.tocsr()
print('Done!')

# Save csc-formatted dataset
indptr = csr.indptr.astype(np.int64)
indices = csr.indices.astype(np.int64)
features = dataset[0].x
labels = dataset[0].y

os.makedirs(dataset_path, exist_ok=True)
indptr_path = os.path.join(dataset_path, 'indptr.dat')
indices_path = os.path.join(dataset_path, 'indices.dat')
features_path = os.path.join(dataset_path, 'features.dat')
labels_path = os.path.join(dataset_path, 'labels.dat')
conf_path = os.path.join(dataset_path, 'conf.json')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')

print('Saving indptr...')
indptr_mmap = np.memmap(indptr_path, mode='w+', shape=indptr.shape, dtype=indptr.dtype)
indptr_mmap[:] = indptr[:]
indptr_mmap.flush()
print('Done!')

print('Saving indices...')
indices_mmap = np.memmap(indices_path, mode='w+', shape=indices.shape, dtype=indices.dtype)
indices_mmap[:] = indices[:]
indices_mmap.flush()
print('Done!')

print('Saving features...')
features_mmap = np.memmap(features_path, mode='w+', shape=(dataset[0].x.shape[0], 1*dataset[0].x.shape[1]), dtype=np.float32)
features_mmap[:] = features[:]
features_mmap.flush()
print('Done!')

print('Saving labels...')
labels = labels.type(torch.float32)
labels_mmap = np.memmap(labels_path, mode='w+', shape=dataset[0].y.shape, dtype=np.float32)
labels_mmap[:] = labels[:]
labels_mmap.flush()
print('Done!')


def default_dump(obj):
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


print('Making conf file...')
mmap_config = dict()
mmap_config['num_nodes'] = dataset[0].num_nodes
mmap_config['indptr_shape'] = tuple(indptr.shape)
mmap_config['indptr_dtype'] = str(indptr.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['features_shape'] = tuple(features_mmap.shape)
mmap_config['features_dtype'] = str(features_mmap.dtype)
mmap_config['labels_shape'] = tuple(labels_mmap.shape)
mmap_config['labels_dtype'] = str(labels_mmap.dtype)
mmap_config['num_classes'] = dataset.num_classes
json.dump(mmap_config, open(conf_path, 'w'), default=default_dump)
print('Done!')

print('Saving split index...')
torch.save(dataset.get_idx_split(), split_idx_path)
print('Done!')
