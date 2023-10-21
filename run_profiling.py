import argparse
import time
import os
import gc
import glob
from datetime import datetime
import torch
import torch.nn.functional as F
from tqdm import tqdm
import threading
from queue import Queue
# from sage import SAGE
from custom_sage import SAGE_PRUNE

from lib.data import *
from lib.cache import *
from lib.utils import *
from lib.neighbor_sampler import PartitionNeighborSampler


# Parse arguments
argparser = argparse.ArgumentParser()
argparser.add_argument('--gpu', type=int, default=0)
argparser.add_argument('--batch-size', type=int, default=1000)
argparser.add_argument('--num-epochs', type=int, default=5)
argparser.add_argument('--num-workers', type=int, default=os.cpu_count()*2)
argparser.add_argument('--num-hiddens', type=int, default=256)
argparser.add_argument('--dataset', type=str, default='mag240m')
argparser.add_argument('--exp-name', type=str, default=None)
argparser.add_argument('--sizes', type=str, default='10,10,10')
argparser.add_argument('--feature-cache-size', type=float, default=30000000000)
argparser.add_argument('--trace-load-num-threads', type=int, default=os.cpu_count())
argparser.add_argument('--neigh-cache-size', type=int, default=10000000000)
argparser.add_argument('--num-threads', type=int, default=os.cpu_count()*8)
argparser.add_argument('--staleness-threshold', type=int, default=100)
argparser.add_argument('--partition-num', type=int, default=10000)
argparser.add_argument('--partition-batch', type=int, default=20)

args = argparser.parse_args()

# Set args/environment variables/path
os.environ['NUM_THREADS'] = str(args.num_threads)
os.environ['STALENESS'] = str(args.staleness_threshold)
dataset_path = os.path.join('./dataset', args.dataset + '-new')
split_idx_path = os.path.join(dataset_path, 'split_idx.pth')
partition_path = f'./fennel_twolevel_{args.partition_num}_part_{args.dataset}'

# Prepare dataset
if args.exp_name is None:
    now = datetime.now()
    args.exp_name = now.strftime('%Y_%m_%d_%H_%M_%S')
os.makedirs(os.path.join('./trace', args.exp_name), exist_ok=True)
sizes = [int(size) for size in args.sizes.split(',')]
dataset = NewDataset(path=dataset_path, split_idx_path=split_idx_path)
num_nodes = dataset.num_nodes
num_features = dataset.num_features
features = dataset.features_path
num_classes = dataset.num_classes
mmapped_features = dataset.get_mmapped_features()
indptr, indices = dataset.get_adj_mat()
labels = dataset.get_labels()

# Define model
device = torch.device('cuda:%d' % args.gpu)
torch.cuda.set_device(device)
model = SAGE_PRUNE(num_features, args.num_hiddens, num_classes, num_layers=len(sizes))
model = model.to(device)


def sampling():
    # import faulthandler; faulthandler.enable()
    # Same effect of `sysctl -w vm.drop_caches=1`
    # Requires sudo
    with open('/proc/sys/vm/drop_caches', 'w') as stream:
        stream.write('1\n')

    st = time.time()
    # Load neighbor cache
    neighbor_cache_path = str(dataset_path) + '/nc' + '_size_' + str(args.neigh_cache_size) + '.dat'
    neighbor_cache_conf_path = str(dataset_path) + '/nc' + '_size_' + str(args.neigh_cache_size) + '_conf.json'
    neighbor_cache_numel = json.load(open(neighbor_cache_conf_path, 'r'))['shape'][0]
    neighbor_cachetable_path = str(dataset_path) + '/nctbl' + '_size_' + str(args.neigh_cache_size) + '.dat'
    neighbor_cachetable_conf_path = str(dataset_path) + '/nctbl' + '_size_' + str(args.neigh_cache_size) + '_conf.json'
    neighbor_cachetable_numel = json.load(open(neighbor_cachetable_conf_path, 'r'))['shape'][0]
    neighbor_cache = load_int64(neighbor_cache_path, neighbor_cache_numel)
    neighbor_cache_ptr = get_tensor_ptr(neighbor_cache)
    neighbor_cachetable = load_int64(neighbor_cachetable_path, neighbor_cachetable_numel)
    neighbor_cachetable_ptr = get_tensor_ptr(neighbor_cachetable)
    cache_load_time = time.time() - st

    loader = PartitionNeighborSampler(indptr, indices, args.exp_name, mode='profiling',
                                        partition_num=args.partition_num, partition_batch=args.partition_batch, partition_path=partition_path,
                                        sizes=sizes, num_nodes = num_nodes,
                                        cache_data = neighbor_cache, address_table = neighbor_cachetable,
                                        shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)
    st = time.time()
    for step, _ in enumerate(loader):
        abc = 123
    sample_time = time.time() - st

    tensor_ptr_free(neighbor_cache_ptr)
    tensor_ptr_free(neighbor_cachetable_ptr)

    num_iter = loader.batch_count

    return loader.access_freq, loader.neigh_size_before, loader.neigh_size_after, sample_time, num_iter


def trace_load(q, indices):
    for i in indices:
        q.put((
            torch.load('./trace/' + args.exp_name + '/' + '_ids_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + '_adjs_' + str(i) + '.pth'),
            ))


def gather_baseline(gather_q, n_id, batch_size):
    s = time.time()

    batch_inputs = gather_direct(features, n_id, num_features)
    # batch_inputs = gather_mmap(mmapped_features, n_id)

    batch_labels = labels[n_id[:batch_size]]
    gather_time = time.time() - s
    gather_q.put((batch_inputs, batch_labels, gather_time))


def delete_trace():
    n_id_filelist = glob.glob('./trace/' + args.exp_name + '/' + '_ids_*')
    adjs_filelist = glob.glob('./trace/' + args.exp_name + '/' + '_adjs_*')
    cache_filelist = glob.glob('./trace/' + args.exp_name + '/' + '_update_*')

    for n_id_file in n_id_filelist:
        try:
            os.remove(n_id_file)
        except:
            tqdm.write('Error while deleting file : ', n_id_file)

    for adjs_file in adjs_filelist:
        try:
            os.remove(adjs_file)
        except:
            tqdm.write('Error while deleting file : ', adjs_file)

    for cache_file in cache_filelist:
        try:
            os.remove(cache_file)
        except:
            tqdm.write('Error while deleting file : ', cache_file)


def execute(pbar, num_iter=0):
    # Multi-threaded load of sets of (ids, adj, update)
    q = list()
    loader = list()
    for t in range(args.trace_load_num_threads):
        q.append(Queue(maxsize=2))
        loader.append(threading.Thread(target=trace_load, args=(q[t], list(range(t, num_iter, args.trace_load_num_threads))), daemon=True))
        loader[t].start()

    gather_q = Queue(maxsize=1)

    total_load_time = 0.
    total_gather_time = 0.
    total_loaded_node_num = 0
    for idx in range(num_iter):
        # for 1 mini-batch
        if idx == 0:
            # Sample
            s = time.time()
            q_value = q[idx % args.trace_load_num_threads].get()
            if q_value:
                n_id, adjs = q_value
                total_loaded_node_num += len(n_id)
                batch_size = adjs[-1].size[1]
                adjs_1 = adjs
                adjs_host = adjs_1
            total_load_time += time.time() - s

            # Gather
            s = time.time()
            batch_inputs = gather_direct(features, n_id, num_features)

            batch_labels = labels[n_id[:batch_size]]
            total_gather_time += time.time() - s

        if idx != 0:
            # Gather
            (batch_inputs, batch_labels, gather_time) = gather_q.get()
            total_gather_time += gather_time

        if idx != num_iter-1:
            # Sample
            s = time.time()
            q_value = q[(idx + 1) % args.trace_load_num_threads].get()
            if q_value:
                n_id, adjs = q_value
                total_loaded_node_num += len(n_id)
                batch_size = adjs[-1].size[1]
                if idx != 0:
                    adjs_host = adjs_2
                adjs_2 = adjs
            total_load_time += time.time() - s

            # Gather
            gather_loader = threading.Thread(target=gather_baseline, args=(gather_q, n_id, batch_size), daemon=True)
            gather_loader.start()

        if idx == num_iter - 1:
            adjs_host = adjs_2

        tensor_free(batch_inputs)
        pbar.update(len(batch_labels))

    return total_gather_time


def profile(epoch):
    num_iter = int(args.partition_num / args.partition_batch)

    pbar = tqdm(total=dataset.train_idx.numel(), position=0, leave=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_profiling_time = 0.
    st = time.time()
    # sampling
    access_freq, neigh_size_before, neigh_size_after, sample_time, num_iter = sampling()

    # gather
    gather_time = execute(pbar, num_iter=num_iter)

    beta = (sample_time/neigh_size_before.sum()) / (gather_time/neigh_size_after.sum())
    node_wise_io = neigh_size_before*beta + neigh_size_after
    node_wise_io_sorted, node_wise_io_idx = node_wise_io.sort(descending=True)
    access_freq_sorted, access_freq_idx = access_freq.sort(descending=True)

    node_wise_io_sorted = torch.cumsum(node_wise_io_sorted, dim=-1)
    access_freq_sorted = torch.cumsum(access_freq_sorted, dim=-1)

    highest_benefit = 0.
    best_alpha = 0.
    benefit_list = []
    for alpha in range(1, 100):
        alpha /= 100
        num_entry_embedding = min(num_nodes, max(0, int((args.feature_cache_size*alpha - num_nodes*4*2) / (args.num_hiddens*4+4))))
        num_entry_feat = min(num_nodes, max(0, int((args.feature_cache_size*(1-alpha) - num_nodes*4) / (num_features*4))))

        benefit = 0.
        if num_entry_embedding > 0:
            benefit += node_wise_io_sorted[num_entry_embedding-1]
        if num_entry_feat > 0:
            benefit += access_freq_sorted[num_entry_feat-1]
        if benefit > highest_benefit:
            highest_benefit = benefit
            best_alpha = alpha
            num_for_embedding = num_entry_embedding
            num_for_feat = num_entry_feat

        benefit_list.append(benefit.item())

    # Delete obsolete runtime files
    delete_trace()

    pbar.close()
    total_profiling_time += time.time() - st

    return node_wise_io_idx[:num_for_embedding], access_freq_idx[:num_for_feat], best_alpha


if __name__=='__main__':
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

    embedding_entry_idx, feat_entry_idx, alpha = profile(epoch=0)

    embedding_entry_path = str(dataset_path) + '/emb_entry.dat'
    feat_entry_path = str(dataset_path) + '/feat_entry.dat'
    alpha_path = str(dataset_path) + '/alpha.dat'
    torch.save(embedding_entry_idx, embedding_entry_path)
    torch.save(feat_entry_idx, feat_entry_path)
    torch.save(alpha, alpha_path)
