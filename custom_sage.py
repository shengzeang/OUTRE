import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch_geometric.nn import SAGEConv

from dgl.utils import pin_memory_inplace, gather_pinned_tensor_rows

from lib.utils import *


class SAGE_PRUNE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE_PRUNE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, x, adjs, n_id, emb_cache_table, true_emb_cache_table, emb_cache_stale, emb_cache, n_iter, device=None, feat_cache=None, hit_idx=None, hit_mask=None):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        # change from DMA-based transfer to zero-copy-based transfer

        hit_feature = gather_pinned_tensor_rows(feat_cache.cache, hit_idx)
        new_feature = torch.zeros((len(hit_mask), x.shape[1]), device=device)
        new_feature[torch.where(hit_mask==True)[0]] = hit_feature
        new_feature[torch.where(hit_mask==False)[0]] = x
        del(hit_feature)
        del(x)
        x = new_feature

        for i, (edge_index, _, size) in enumerate(adjs):
            if i == self.num_layers - 1:
                push_batch_id, push_global_id, pull_batch_id, pull_global_id = cache_check(n_id, emb_cache_table, true_emb_cache_table, emb_cache_stale, n_iter)
                # push to cache
                emb_cache[emb_cache_table[push_global_id].long()] = x[push_batch_id].detach().cpu()
                # pull from cache
                x[pull_batch_id] = emb_cache[emb_cache_table[pull_global_id].long()].to(x.device)

                # print(len(pull_global_id), len(push_global_id))

                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
            else:
                x_target = x[:size[1]]  # Target nodes are always placed first.
                x = self.convs[i]((x, x_target), edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
            n_id = n_id[:size[1]]
        return x.log_softmax(dim=-1)


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(SAGE, self).__init__()

        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()


    def forward(self, adjs, x_tar, x_neigh):
        # `train_loader` computes the k-hop neighborhood of a batch of nodes,
        # and returns, for each layer, a bipartite graph object, holding the
        # bipartite edges `edge_index`, the index `e_id` of the original edges,
        # and the size/shape `size` of the bipartite graph.
        # Target nodes are also included in the source nodes so that one can
        # easily apply skip-connections or add self-loops.

        x = self.convs[0]((x_neigh, x_tar))

        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i+1]((x, x_target), edge_index)
            if i + 1 != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)

        return x.log_softmax(dim=-1)
