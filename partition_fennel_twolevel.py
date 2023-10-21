import os
import time
import gc
import argparse
import json
os.environ['DGLBACKEND'] = 'pytorch'


import torch
import scipy
import numpy as np

import dgl
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from dgl.data import AmazonCoBuyComputerDataset, AmazonCoBuyPhotoDataset
from dgl.data import CoauthorCSDataset, CoauthorPhysicsDataset

from lib.utils import *
from lib.data import *


argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='mag240m')
argparser.add_argument('--num-threads', type=int, default=int(os.cpu_count()))
argparser.add_argument('--num-rounds', type=int, default=1)
argparser.add_argument('--num-partitions', type=int, default=10000)
argparser.add_argument('--first-level-num', type=int, default=100)
argparser.add_argument('--imbalance-ratio', type=float, default=1.1)
argparser.add_argument('--gamma', type=float, default=2.0)
argparser.add_argument('--patience', type=int, default=3)
args = argparser.parse_args()
os.environ['NUM_THREADS'] = str(args.num_threads)

if args.dataset == 'Cora':
    dataset = CoraGraphDataset()
    graph = dataset[0]
elif args.dataset == 'Citeseer':
    dataset = CiteseerGraphDataset()
    graph = dataset[0]
elif args.dataset == 'PubMed':
    dataset = PubmedGraphDataset()
    graph = dataset[0]
elif args.dataset == 'Computers':
    dataset = AmazonCoBuyComputerDataset()
    graph = dataset[0]
elif args.dataset == 'Photo':
    dataset = AmazonCoBuyPhotoDataset()
    graph = dataset[0]
elif args.dataset == 'CS':
    dataset = CoauthorCSDataset()
    graph = dataset[0]
elif args.dataset == 'Physics':
    dataset = CoauthorPhysicsDataset()
    graph = dataset[0]
elif args.dataset in ['ogbn-arxiv', 'ogbn-products', 'ogbn-papers100M', 'mag240m', 'igb-small', 'igb-medium', 'igb-large']:
    dataset_path = os.path.join('./dataset', args.dataset + '-new')
    split_idx_path = os.path.join(dataset_path, 'split_idx.pth')
    dataset = NewDataset(path=dataset_path, split_idx_path=split_idx_path)

    num_nodes = dataset.num_nodes
    num_features = dataset.num_features
    features = dataset.features_path
    num_classes = dataset.num_classes

    if args.dataset != 'mag240m':
        train_nid, val_nid, test_nid = dataset.train_idx, dataset.val_idx, dataset.test_idx
        num_train_nodes = train_nid.shape[0]
        num_val_nodes = val_nid.shape[0]
        num_test_nodes = test_nid.shape[0]
    else:
        train_nid, val_nid = dataset.train_idx, dataset.val_idx
        num_train_nodes = train_nid.shape[0]
        num_val_nodes = val_nid.shape[0]

    train_mask = torch.zeros((num_nodes,), dtype=torch.bool)
    train_mask[train_nid] = True
    val_mask = torch.zeros((num_nodes,), dtype=torch.bool)
    val_mask[val_nid] = True
    if args.dataset != 'mag240m':
        test_mask = torch.zeros((num_nodes,), dtype=torch.bool)
        test_mask[test_nid] = True

    del(dataset)
    gc.collect()
else:
    assert(False)

partition_num = args.first_level_num
second_level_num = int(args.num_partitions / args.first_level_num)
out_path = f'./fennel_twolevel_{args.num_partitions}_part_{args.dataset}'
if os.path.exists(out_path) == False:
    os.mkdir(out_path)
for part in range(args.num_partitions):
    if os.path.exists(out_path+f"/part{part}") == False:
        os.mkdir(out_path+f"/part{part}")

indptr_path = os.path.join(dataset_path, 'indptr.dat')
indices_path = os.path.join(dataset_path, 'indices.dat')
conf_path = os.path.join(dataset_path, 'conf.json')
conf = json.load(open(conf_path, 'r'))

indptr_size = conf['indptr_shape'][0]
csr_indptr = load_int64(indptr_path, indptr_size)
num_nodes = csr_indptr.shape[0] - 1
indices_size = conf['indices_shape'][0]
csr_indices = load_int64(indices_path, indices_size)
num_edges = csr_indices.shape[0]

with open('/proc/sys/vm/drop_caches', 'w') as stream:
    stream.write('1\n')

max_size = int(float(num_nodes) / float(partition_num) * args.imbalance_ratio)
result = torch.full((partition_num, max_size), -1).long()

cross_edge_num = torch.zeros(num_nodes).long()

st = time.time()
fennel_partition(result, cross_edge_num, csr_indptr, csr_indices, partition_num, args.num_rounds,
                  num_train_nodes, train_mask, args.imbalance_ratio, args.gamma, args.patience, torch.arange(num_nodes).long())
'''fennel_bf_partition(result, cross_edge_num, csr_indptr, csr_indices, partition_num, args.num_rounds,
                  num_train_nodes, train_mask, args.imbalance_ratio, args.gamma, args.patience, torch.arange(num_nodes).long())'''

'''del(csr_indices)
gc.collect()
fennel_bf_partition_outofcore(result, cross_edge_num, csr_indptr, indices_path, num_edges, partition_num, args.num_rounds,
                  num_train_nodes, train_mask, args.imbalance_ratio, args.gamma, args.patience, torch.arange(num_nodes).long())'''

for part in range(partition_num):
    inpart_nodes = result[part]
    inpart_nodes = inpart_nodes[inpart_nodes!=-1]
    torch.save(inpart_nodes, out_path+f"/part{part*second_level_num}/n_id.dat")

    train_nid_part = fetch_split_nodes(inpart_nodes, train_mask)
    torch.save(train_nid_part, out_path+f"/part{part*second_level_num}/train_n_id.dat")
    val_nid_part = fetch_split_nodes(inpart_nodes, val_mask)
    torch.save(val_nid_part, out_path+f"/part{part*second_level_num}/val_n_id.dat")
    if args.dataset != 'mag240m':
        test_nid_part = fetch_split_nodes(inpart_nodes, test_mask)
        torch.save(test_nid_part, out_path+f"/part{part*second_level_num}/test_n_id.dat")

    # indptr, indices, local_global_mapping = fetch_csr_first_level_outofcore(csr_indptr, indices_path, inpart_nodes)
    indptr, indices, local_global_mapping = fetch_csr_first_level(csr_indptr, csr_indices, inpart_nodes)

    torch.save(local_global_mapping.long(), out_path+f"/part{part*second_level_num}/mapping.dat")
    torch.save(indptr.long(), out_path+f"/part{part*second_level_num}/csr_indptr.dat")
    torch.save(indices.long(), out_path+f"/part{part*second_level_num}/csr_indices.dat")
print(f'First-level graph partition takes {np.round(time.time() - st, 2)}s')

# after first-level partition, map the node ids to local id, and store the mapping
# after second-level partition, map the node ids back to global id, using the mapping store before
cross_edge_num = torch.zeros(num_nodes).long() - 1
for sub_part in range(args.first_level_num):
    partition_num = second_level_num

    csr_indptr_part = torch.load(out_path+f"/part{sub_part*second_level_num}/csr_indptr.dat")
    num_nodes_part = csr_indptr_part.shape[0]- 1
    csr_indices_part = torch.load(out_path+f"/part{sub_part*second_level_num}/csr_indices.dat")
    num_edges_part = csr_indices_part.shape[0]
    train_nid_part = torch.load(out_path+f"/part{sub_part*second_level_num}/train_n_id.dat")
    num_train_nodes_part = train_nid.shape[0]
    train_mask_part = train_mask[torch.load(out_path+f"/part{sub_part*second_level_num}/n_id.dat")]
    local_to_global_mapping_part = torch.load(out_path+f"/part{sub_part*second_level_num}/mapping.dat")

    max_size_part = int(float(num_nodes_part) / float(partition_num) * args.imbalance_ratio)
    result_part = torch.full((partition_num, max_size_part), -1).long()

    fennel_partition(result_part, cross_edge_num, csr_indptr_part, csr_indices_part, partition_num, args.num_rounds,
                    num_train_nodes_part, train_mask_part, args.imbalance_ratio, args.gamma, args.patience, local_to_global_mapping_part)
    '''fennel_bf_partition(result_part, cross_edge_num, csr_indptr_part, csr_indices_part, partition_num, args.num_rounds,
                    num_train_nodes_part, train_mask_part, args.imbalance_ratio, args.gamma, args.patience, local_to_global_mapping_part)'''
    print(f"done for second-level part {sub_part}")

    local_global_mapping = torch.load(out_path+f"/part{sub_part*second_level_num}/mapping.dat")
    for part in range(partition_num):
        inpart_nodes = result_part[part]
        inpart_nodes = inpart_nodes[inpart_nodes!=-1]
        # transform to global ID
        inpart_nodes = local_global_mapping[inpart_nodes]
        torch.save(inpart_nodes, out_path+f"/part{sub_part*second_level_num+part}/n_id.dat")

        train_nid_part = fetch_split_nodes(inpart_nodes, train_mask)
        torch.save(train_nid_part, out_path+f"/part{sub_part*second_level_num+part}/train_n_id.dat")
        val_nid_part = fetch_split_nodes(inpart_nodes, val_mask)
        torch.save(val_nid_part, out_path+f"/part{sub_part*second_level_num+part}/val_n_id.dat")
        if args.dataset != 'mag240m':
            test_nid_part = fetch_split_nodes(inpart_nodes, test_mask)
            torch.save(test_nid_part, out_path+f"/part{sub_part*second_level_num+part}/test_n_id.dat")

        # indptr, indices = fetch_csr_outofcore(csr_indptr, indices_path, inpart_nodes)
        indptr, indices = fetch_csr(csr_indptr, csr_indices, inpart_nodes)

        torch.save(indptr.long(), out_path+f"/part{sub_part*second_level_num+part}/csr_indptr.dat")
        torch.save(indices.long(), out_path+f"/part{sub_part*second_level_num+part}/csr_indices.dat")


score_path = f'./dataset/{args.dataset}-new/nc_score.pth'
torch.save(cross_edge_num, score_path)

print(f'The whole graph partition takes {np.round(time.time() - st, 2)}s')
