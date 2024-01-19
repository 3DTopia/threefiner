import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import tinycudann as tcnn

class HashGridEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 num_levels=16,
                 level_dim=2,
                 log2_hashmap_size=18, 
                 base_resolution=16, 
                 desired_resolution=1024, 
                 interpolation='linear'
                 ):
        super().__init__()
        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": np.exp2(np.log2(desired_resolution / num_levels) / (num_levels - 1)),
                "interpolation": "Smoothstep" if interpolation == 'smoothstep' else "Linear",
            },
            dtype=torch.float32,
        )
        self.input_dim = input_dim
        self.output_dim = self.encoder.n_output_dims # patch
    
    def forward(self, x, bound=1):
        return self.encoder((x + bound) / (2 * bound))

class FrequencyEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 output_dim=32,
                 n_frequencies=12,
                 ):
        super().__init__()
        self.encoder = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": n_frequencies,
            },
            dtype=torch.float32,
        )
        self.implicit_mlp = MLP(self.encoder.n_output_dims, output_dim, 128, 5, bias=True)
        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x, **kwargs):
        return self.implicit_mlp(self.encoder(x))
    

class TriplaneEncoder(nn.Module):
    def __init__(self, 
                 input_dim=3,
                 output_dim=32,
                 resolution=256,
                 ):
        super().__init__()

        self.C_mat = nn.Parameter(torch.randn(3, output_dim, resolution, resolution))
        torch.nn.init.kaiming_normal_(self.C_mat)
        
        self.mat_ids = [[0, 1], [0, 2], [1, 2]]

        self.input_dim = input_dim
        self.output_dim = output_dim
    
    def forward(self, x, bound=1):

        N = x.shape[0]
        x = x / bound # to [-1, 1]

        mat_coord = torch.stack((x[..., self.mat_ids[0]], x[..., self.mat_ids[1]], x[..., self.mat_ids[2]])).view(3, -1, 1, 2) # [3, N, 1, 2]

        feat = F.grid_sample(self.C_mat[[0]], mat_coord[[0]], align_corners=False).view(-1, N) + \
               F.grid_sample(self.C_mat[[1]], mat_coord[[1]], align_corners=False).view(-1, N) + \
               F.grid_sample(self.C_mat[[2]], mat_coord[[2]], align_corners=False).view(-1, N) # [r, N]

        # density
        feat = feat.transpose(0, 1).contiguous() # [N, C]
        return feat

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        net = []
        for l in range(num_layers):
            net.append(nn.Linear(self.dim_in if l == 0 else self.dim_hidden, self.dim_out if l == num_layers - 1 else self.dim_hidden, bias=bias))

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                x = F.relu(x, inplace=True)
        return x