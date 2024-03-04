import functools as ft

import torch as th
import torch.nn.functional as F
import torch.nn as nn 

from utilities.utils import *

class MLP_Model(nn.Module):
    def __init__(self, layer_cfg, non_linears, dropouts):
        super(MLP_Model, self).__init__()
        self.shapes = zip(layer_cfg[:-1], layer_cfg[1:])
        self.non_linears = non_linears 
        self.dropouts = dropouts 

        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.Dropout(p=val),
                nn.ReLU() if apply_fn == 1 else nn.Identity()
            )
            for (in_features, out_features), apply_fn, val in zip(self.shapes, self.non_linears, self.dropouts)
        ])
        
    def forward(self, X0: th.FloatTensor):
        XN = ft.reduce(lambda Xi, Li: Li(Xi), self.layers, X0)
        return XN

class CNN_Model(nn.Module):
    def __init__(self, layer_cfg, stride=1, padding=2, conv_kernel_size=5, pool_kernel_size=2):
        super(CNN_Model, self).__init__()
        
        conv_shapes = layer_cfg['convs']['shapes']
        conv_dropouts = layer_cfg['convs']['dropouts']
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel_size, stride=stride, padding=padding),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=pool_kernel_size),
                nn.Dropout(p=dropout)
            )
            for (in_channels, out_channels), dropout in zip(zip(conv_shapes[:-1], conv_shapes[1:]), conv_dropouts)
        ])

        linear_shapes = layer_cfg['linears']['shapes']
        non_linears = layer_cfg['linears']['non_linears']
        linear_dropouts = layer_cfg['linears']['dropouts']
        self.linear_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.ReLU() if apply_fn == 1 else nn.Identity(),
                nn.Dropout(p=val)
            )
            for (in_features, out_features), apply_fn, val in zip(
                zip(linear_shapes[:-1], linear_shapes[1:]), non_linears, linear_dropouts
            )
        ])

    def forward(self, X0: th.FloatTensor):
        X_conv = ft.reduce(lambda X, layer: layer(X), self.conv_layers, X0)
        X_flatten = X_conv.view(X_conv.size(0), -1)
        XN = ft.reduce(lambda X, layer: layer(X), self.linear_layers, X_flatten)
        return XN

if __name__ == '__main__':
    logger.info('Model testing...')
    model = MLP_Model(layer_cfg=[384, 128, 64, 10],  non_linears=[1, 1, 0], dropouts=[0.3, 0.2, 0.0])
    print(model)
    # X = th.randn((10, 384))
    # O = model(X)
    # print(O)
    
    layer_cfg = {
        'convs': {
            'shapes': [1, 32, 64],
            'dropouts': [0.25, 0.5]
        },
        'linears': {
            'shapes': [1024, 10],
            'non_linears': [1, 0],
            'dropouts': [0.5, 0.0]
        }
    }
    
    model = CNN_Model(layer_cfg)
    print(model)