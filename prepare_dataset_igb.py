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
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='./dataset',
    help='path containing the datasets')
parser.add_argument('--dataset_size', type=str, default='medium',
    choices=['tiny', 'small', 'medium', 'large', 'full'],
    help='size of the datasets')
parser.add_argument('--num_classes', type=int, default=19,
    choices=[19, 2983], help='number of classes')
parser.add_argument('--in_memory', type=int, default=0,
    choices=[0, 1], help='0:read only mmap_mode=r, 1:load into memory')
parser.add_argument('--synthetic', type=int, default=0,
    choices=[0, 1], help='0:nlp-node embeddings, 1:random')
args = parser.parse_args()

# Download/load dataset
print('Loading dataset...')
root = './dataset/'
os.makedirs(root, exist_ok=True)
from igb import download
download.download_dataset(path=args.path, dataset_type='homogeneous', dataset_size=args.dataset_size)

from igb.dataloader import IGB260MDGLDataset
dataset = IGB260MDGLDataset(args)
graph = dataset[0]
dataset_path = os.path.join(root, 'igb-' + args.dataset_size + '-new')
graph.ndata['label'] = graph.ndata['label'].unsqueeze(1)
edge_index = torch.cat((graph.edges()[0].unsqueeze(0), graph.edges()[1].unsqueeze(0)), dim=0)
dataset_split = {}
dataset_split['train'] = torch.where(graph.ndata['train_mask']==True)[0]
dataset_split['valid'] = torch.where(graph.ndata['val_mask']==True)[0]
dataset_split['test'] = torch.where(graph.ndata['test_mask']==True)[0]

# Construct sparse formats
print('Creating coo/csc/csr format of dataset...')
num_nodes = graph.num_nodes()
coo = to_undirected(edge_index).numpy()
v = np.ones_like(coo[0])
coo = scipy.sparse.coo_matrix((v, (coo[0], coo[1])), shape=(num_nodes, num_nodes))
csc = coo.tocsc()
csr = coo.tocsr()
print('Done!')

# Save csc-formatted dataset
indptr = csr.indptr.astype(np.int64)
indices = csr.indices.astype(np.int64)
features = graph.ndata['feat']
labels = graph.ndata['label']

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
features_mmap = np.memmap(features_path, mode='w+', shape=graph.ndata['feat'].shape, dtype=np.float32)
step = 1000000
for i in range(int(num_nodes.step)):
    features_mmap[i*step:min((i+1)*step, num_nodes)] = features[i*step:min((i+1)*step, num_nodes)]
    features_mmap.flush()
print('Done!')

print('Saving labels...')
labels = labels.type(torch.float32)
labels_mmap = np.memmap(labels_path, mode='w+', shape=graph.ndata['label'].shape, dtype=np.float32)
labels_mmap[:] = labels[:]
labels_mmap.flush()
print('Done!')

print('Making conf file...')
mmap_config = dict()
mmap_config['num_nodes'] = graph.num_nodes()
mmap_config['indptr_shape'] = tuple(indptr.shape)
mmap_config['indptr_dtype'] = str(indptr.dtype)
mmap_config['indices_shape'] = tuple(indices.shape)
mmap_config['indices_dtype'] = str(indices.dtype)
mmap_config['features_shape'] = tuple(features_mmap.shape)
mmap_config['features_dtype'] = str(features_mmap.dtype)
mmap_config['labels_shape'] = tuple(labels_mmap.shape)
mmap_config['labels_dtype'] = str(labels_mmap.dtype)
mmap_config['num_classes'] = args.num_classes
json.dump(mmap_config, open(conf_path, 'w'))
print('Done!')

print('Saving split index...')
torch.save(dataset_split, split_idx_path)
print('Done!')
