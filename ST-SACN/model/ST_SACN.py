import torch
import torch.nn as nn
import torch.nn.functional as F

from function.functions import Chomp1d, sample_neighbors


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=True)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1)

        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        return self.relu(out)


class ST_SACN_Block(nn.Module):
    def __init__(self, in_feats, out_feats, hid_feats, kernel_size, dilation_size, dropout, agg_func='mean'):
        super(ST_SACN_Block, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.agg_func = agg_func
        self.time1 = TemporalBlock(in_feats, out_feats, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size - 1) * dilation_size)
        self.fc = nn.Linear(out_feats * 2, hid_feats)
        self.time2 = TemporalBlock(hid_feats, out_feats, kernel_size, stride=1, dilation=dilation_size,
                                   padding=(kernel_size - 1) * dilation_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_matrices, num_samples, batch_nodes):

        x = x.permute(0, 2, 1)
        x = self.time1(x)
        x = x.permute(0, 2, 1)
        num_nodes, num_timesteps, num_feats = x.shape
        agg_neighbors = torch.zeros((num_nodes, num_timesteps, num_feats)).to(x.device)

        batch_node_set = set(batch_nodes)
        for t in range(num_timesteps):
            sampled_neighbors = sample_neighbors(adj_matrices, batch_node_set, num_samples, t)
            for i, node in enumerate(batch_nodes):
                neighbors = sampled_neighbors[node]
                valid_neighbors = [n for n in neighbors if n in batch_node_set]
                if len(valid_neighbors) > 0:
                    neighbor_feats = torch.stack([x[list(batch_node_set).index(n), t] for n in valid_neighbors], dim=0)
                    if self.agg_func == 'mean':
                        agg_neighbors[i, t] = neighbor_feats.mean(dim=0)
                    elif self.agg_func == 'sum':
                        agg_neighbors[i, t] = neighbor_feats.sum(dim=0)
                    elif self.agg_func == 'max':
                        agg_neighbors[i, t] = neighbor_feats.max(dim=0)[0]
                else:
                    agg_neighbors[i, t] = x[list(batch_node_set).index(node), t]

        batch_feats = x[torch.tensor([list(batch_node_set).index(n) for n in list(batch_node_set)])]
        combined = torch.cat([batch_feats, agg_neighbors], dim=2)
        out = self.fc(combined)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.time2(out)
        out = out.permute(0, 2, 1)
        out = self.dropout(out)
        return out


class ST_SACN(nn.Module):
    def __init__(self, in_feats, out_feats, hidden_feats, num_layers, time_steps, classes):
        super(ST_SACN, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(
            ST_SACN_Block(in_feats, out_feats, hidden_feats, kernel_size=3, dilation_size=1, dropout=0.2))
        for _ in range(num_layers - 2):
            self.layers.append(
                ST_SACN_Block(out_feats, out_feats, hidden_feats, kernel_size=3, dilation_size=1, dropout=0.2))
        self.layers.append(
            ST_SACN_Block(out_feats, out_feats, hidden_feats, kernel_size=3, dilation_size=1, dropout=0.2))
        self.fc = nn.Linear(out_feats * time_steps, classes)

    def forward(self, x, adj_matrices, num_samples_list, batch_nodes):
        for layer, num_samples in zip(self.layers, num_samples_list):
            x = layer(x, adj_matrices, num_samples, batch_nodes)
        x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        out = self.fc(x)
        return out
