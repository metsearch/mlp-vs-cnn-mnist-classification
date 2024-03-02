import functools as ft

import torch as th 
import torch.nn as nn 

from utilities.utils import *

class MLP_Model(nn.Module):
    def __init__(self, layer_cfg, non_linears, dropouts):
        super(MLP_Model, self).__init__()
        self.shapes = zip(layer_cfg[:-1], layer_cfg[1:])
        self.non_linears = non_linears 
        self.dropouts = dropouts 

        self.layers = nn.ModuleList([])

        for (i_dim, o_dim), apply_fn, val in zip(self.shapes, self.non_linears, self.dropouts):
            linear = nn.Linear(i_dim, o_dim)
            proba = nn.Dropout(p=val)
            theta = nn.ReLU() if apply_fn == 1 else nn.Identity()
            block = nn.Sequential(linear, proba, theta)
            self.layers.append(block)
    
    def forward(self, X0: th.FloatTensor):
        XN = ft.reduce(lambda Xi, Li: Li(Xi), self.layers, X0)
        return XN
    
class CNN_Model(nn.Module):
    def __init__(self, layer_cfg, non_linears, dropouts):
        super(CNN_Model, self).__init__()
        pass
    
    def forward(self, X0: th.FloatTensor):
        pass

if __name__ == '__main__':
    logger.info('Model testing...')
    model = MLP_Model(layer_cfg=[384, 128, 64, 10],  non_linears=[1, 1, 0], dropouts=[0.3, 0.2, 0.0])
    print(model)
    X = th.randn((10, 384))
    O = model(X)
    print(O)