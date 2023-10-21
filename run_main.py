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

from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows

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


def sampling(embedding_entry_idx, mode='train'):
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

    if mode == 'train':
        # initialize the embedding cache
        emb_cache_table = torch.full((num_nodes,), -3).int()
        emb_cache_table[embedding_entry_idx] = torch.full((len(embedding_entry_idx),), -2).int()

        true_emb_cache_table = torch.full((num_nodes,), -3).int()
        true_emb_cache_table[embedding_entry_idx] = torch.arange(len(embedding_entry_idx)).int()

        emb_cache_stale = torch.zeros((len(embedding_entry_idx), )).int()

        emb_cache_path = str(dataset_path) + '/emb_cache.dat'
        emb_cache = torch.zeros((len(embedding_entry_idx), args.num_hiddens))
        torch.save(emb_cache, emb_cache_path)
        del(emb_cache)

        loader = PartitionNeighborSampler(indptr, indices, args.exp_name, mode=mode,
                                        partition_num=args.partition_num, partition_batch=args.partition_batch, partition_path=partition_path,
                                        sizes=sizes, num_nodes = num_nodes,
                                        emb_cache_table=emb_cache_table, true_emb_cache_table=true_emb_cache_table, emb_cache_stale=emb_cache_stale,
                                        cache_data = neighbor_cache, address_table = neighbor_cachetable,
                                        shuffle=False, num_workers=args.num_workers, batch_size=args.batch_size)
    else:
        loader = PartitionNeighborSampler(indptr, indices, args.exp_name, mode=mode,
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

    if mode == 'train':
        return cache_load_time, emb_cache_table, true_emb_cache_table, num_iter
    else:
        return cache_load_time, num_iter


def trace_load(q, indices):
    for i in indices:
        q.put((
            torch.load('./trace/' + args.exp_name + '/' + '_ids_' + str(i) + '.pth'),
            torch.load('./trace/' + args.exp_name + '/' + '_adjs_' + str(i) + '.pth'),
            ))


def init_gather_zerocopy(gather_q, n_id, cache, batch_size):
    s = time.time()
    batch_inputs, hit_num, hit_idx, hit_pos = gather_zerocopy(features, n_id, num_features, cache)
    hit_mask = torch.zeros((len(n_id),), dtype=torch.bool)
    hit_mask[hit_pos] = True

    batch_labels = labels[n_id[:batch_size]]
    gather_time = time.time() - s

    # feature cache update
    s = time.time()
    cache.update_cache(n_id[torch.where(hit_mask==False)[0]], batch_inputs)
    update_time = time.time() - s

    gather_q.put((batch_inputs, batch_labels, gather_time, update_time, hit_num, hit_idx, hit_mask))


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


def execute(feat_cache, pbar, total_loss, total_correct, emb_cache_table=None, true_emb_cache_table=None, mode='train', num_iter=0):
    # Multi-threaded load of sets of (ids, adj, update)
    q = list()
    loader = list()
    for t in range(args.trace_load_num_threads):
        q.append(Queue(maxsize=2))
        loader.append(threading.Thread(target=trace_load, args=(q[t], list(range(t, num_iter, args.trace_load_num_threads))), daemon=True))
        loader[t].start()

    emb_cache_path = str(dataset_path) + '/emb_cache.dat'
    emb_cache = torch.load(emb_cache_path)

    emb_cache_stale = torch.zeros((emb_cache.shape[0], )).int()

    gather_q = Queue(maxsize=1)

    feat_cache.cache = torch.zeros((feat_cache.num_entries, feat_cache.feature_dim), dtype=torch.float32)
    _ = pin_memory_inplace(feat_cache.cache)

    total_load_time = 0.
    total_gather_time = 0.
    total_transfer_time = 0.
    total_for_back_time = 0.
    total_free_time = 0.
    total_cache_update_time = 0.
    total_loaded_node_num = 0
    total_hit_num = 0
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
                # adjs_q.put(adjs)
            total_load_time += time.time() - s

            # Gather
            s = time.time()
            batch_inputs, hit_num, hit_idx, hit_pos = gather_zerocopy(features, n_id, num_features, feat_cache)
            hit_mask = torch.zeros((len(n_id),), dtype=torch.bool)
            hit_mask[hit_pos] = True
            total_hit_num += hit_num

            batch_labels = labels[n_id[:batch_size]]
            total_gather_time += time.time() - s

            # feature cache update
            s = time.time()
            feat_cache.update_cache(n_id[torch.where(hit_mask==False)[0]], batch_inputs)
            total_cache_update_time += time.time() - s

        if idx != 0:
            # Gather
            (batch_inputs, batch_labels, gather_time, update_time, hit_num, hit_idx, hit_mask) = gather_q.get()
            total_cache_update_time += update_time
            total_hit_num += hit_num
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
            gather_loader = threading.Thread(target=init_gather_zerocopy, args=(gather_q, n_id, feat_cache, batch_size), daemon=True)
            gather_loader.start()

        if idx == num_iter - 1:
            adjs_host = adjs_2

        # Transfer
        s = time.time()
        batch_inputs_cuda = batch_inputs.to(device)
        batch_labels_cuda = batch_labels.to(device)
        adjs_cuda = [adj.to(device) for adj in adjs_host]
        hit_idx = hit_idx.to(device)
        hit_mask = hit_mask.to(device)
        total_transfer_time += time.time() - s

        # change from DMA-based transfer to zero-copy-based transfer
        # Forward
        s = time.time()
        out = model(batch_inputs_cuda, adjs_cuda, n_id, emb_cache_table, true_emb_cache_table, emb_cache_stale, emb_cache, idx, device, feat_cache, hit_idx, hit_mask)
        loss = F.nll_loss(out, batch_labels_cuda.long())

        # Backward
        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_for_back_time += time.time() - s

        # Free
        s = time.time()
        total_loss += float(loss)
        total_correct += int(out.argmax(dim=-1).eq(batch_labels_cuda.long()).sum())
        tensor_free(batch_inputs)
        pbar.update(len(batch_labels))
        total_free_time += time.time() - s

    return total_loss, total_correct


def train(epoch, feat_cache, embedding_entry_idx):
    model.train()

    shuffled_partition_idx = torch.randperm(args.partition_num)

    dataset.make_new_shuffled_train_idx()
    num_iter = int(args.partition_num / args.partition_batch)

    pbar = tqdm(total=dataset.train_idx.numel(), position=0, leave=True)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    total_neigh_cache_load_time = 0.
    total_preprocessing_time = 0.
    total_training_time = 0.

    st = time.time()
    # sampling
    cache_load_time, emb_cache_table, true_emb_cache_table, num_iter = sampling(embedding_entry_idx, mode='train')
    total_neigh_cache_load_time += cache_load_time

    ed = time.time()
    total_preprocessing_time += ed - st

    # gather, transfer, and compute
    total_loss, total_correct = execute(feat_cache, pbar, total_loss, total_correct, emb_cache_table, true_emb_cache_table, mode='train', num_iter=num_iter)
    total_training_time += time.time() - ed

    # Delete obsolete runtime files
    delete_trace()

    pbar.close()
    loss = total_loss / num_iter
    approx_acc = total_correct / dataset.train_idx.numel()

    return loss, approx_acc


if __name__=='__main__':
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    embedding_entry_path = str(dataset_path) + '/emb_entry.dat'
    feat_entry_path = str(dataset_path) + '/feat_entry.dat'
    alpha_path = str(dataset_path) + '/alpha.dat'
    embedding_entry_idx = torch.load(embedding_entry_path)
    feat_entry_idx = torch.load(feat_entry_path)
    alpha = torch.load(alpha_path)

    for epoch in range(1, args.num_epochs+1):
        # initialize the feature cache
        feat_cache = FeatureCache(int(args.feature_cache_size*(1-alpha)), num_nodes, mmapped_features, num_features)
        feat_cache.init_cache(feat_entry_idx)

        start = time.time()
        loss, acc = train(epoch, feat_cache, embedding_entry_idx)
        end = time.time()
        tqdm.write(f'Epoch {epoch:02d}, Loss: {loss.item():.4f}, Approx. Train: {acc:.4f}')
        tqdm.write('Epoch time: {:.3f} s'.format((end - start)))
