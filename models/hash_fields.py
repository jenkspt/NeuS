from itertools import chain
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.embedder import get_embedder

from multires_hash_encoding.modules import (
    MultiresEncoding,
    MultiresEncodingConfig,
    ViewEncoding,
    MLP,
)


class SDFNetwork(nn.Sequential):
    def __init__(self,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers):
        
        multires_config = MultiresEncodingConfig()
        encoding_size = multires_config.features * multires_config.nlevels
        super().__init__(
            MultiresEncoding(**vars(multires_config)),
            MLP(encoding_size, *((d_hidden,)*(n_layers-1)), d_out, activation=nn.Softplus(beta=100)),
        )

    def sdf(self, x):
        return self.forward(x)[:, :1]

    def sdf_hidden_appearance(self, x):
        return self.forward(x)

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)


class RenderingNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out,
                 d_hidden,
                 n_layers,):
        super().__init__()
        self.view_encoder = ViewEncoding(4)
        self.normal_encoder = ViewEncoding(4)
        mlp_in = d_feature + self.view_encoder.encoding_size + self.normal_encoder.encoding_size
        self.mlp = MLP(mlp_in, *((d_hidden,)*(n_layers-1)), d_out)

    def forward(self, points, normals, view_dirs, feature_vectors):
        normals = self.normal_encoder(normals)
        view_dirs = self.view_encoder(view_dirs)

        h = torch.cat([feature_vectors, normals, view_dirs], -1)
        h = self.mlp(h)
        h = torch.sigmoid(h)
        return h


class NeRF(nn.Module):
    def __init__(self,
                 D=3,
                 W=64,
                 d_in=3,
                 d_in_view=3,):
        super().__init__()
        #assert (d_in == 3) and (d_in_view == 3)
        multires_config = MultiresEncodingConfig()
        encoding_size = multires_config.features * multires_config.nlevels
        self.feature_net = nn.Sequential(
            MultiresEncoding(**vars(multires_config)),
            MLP(encoding_size, *((W,)*D)),
            nn.ReLU(),)
        self.alpha_linear = nn.Linear(W, 1)
        self.view_encoder = ViewEncoding(4)
        self.rgb_mlp = MLP(W + self.view_encoder.encoding_size, W, 3)

    def forward(self, input_pts, input_views):
        h = self.feature_net(input_pts[:, :3])
        alpha = self.alpha_linear(h)
        h = torch.cat([self.view_encoder(input_views), h], -1)
        rgb = self.rgb_mlp(h)
        return alpha, rgb


class SingleVarianceNetwork(nn.Module):
    def __init__(self, init_val):
        super().__init__()
        self.variance = nn.Parameter(torch.tensor(init_val))

    def forward(self, x):
        return torch.ones([len(x), 1], dtype=self.variance.dtype, device=self.variance.device) * torch.exp(self.variance * 10.0)
