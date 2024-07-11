import torch.nn as nn
import random


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


def sample_neighbors(adj_matrices, nodes, num_samples, time_step):
    sampled_neighbors = {}
    adj_matrix = adj_matrices[time_step]
    for node in nodes:
        neighbors = adj_matrix[node].nonzero(as_tuple=False).squeeze().tolist()
        if isinstance(neighbors, int):
            neighbors = [neighbors]
        if len(neighbors) < num_samples:
            sampled_neighbors[node] = neighbors
        else:
            sampled_neighbors[node] = random.sample(neighbors, num_samples)
    return sampled_neighbors
