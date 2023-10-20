import numpy as np

import torch
from torch import nn


class LinearReLUBlock(nn.Module):
    def __init__(self, n_in:int, n_out:int)->None:
        super().__init__()
        
        self.stack = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.LeakyReLU(),
            nn.BatchNorm1d(n_out),
        )
        
    def forward(self, x):
        return self.stack(x)

class VanillaNN(nn.Module):
    def __init__(
            self, 
            n_hidden:int = 128, 
        )->None:
        super().__init__()
        self.blk1 = LinearReLUBlock(2, n_hidden)
        self.blk2 = LinearReLUBlock(n_hidden, n_hidden)
        self.blk3 = LinearReLUBlock(n_hidden, n_hidden)
        self.output = nn.Sequential(
            nn.Linear(n_hidden, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x1 = self.blk1(x)
        x2 = x1 + self.blk2(x1)
        x3 = x1 + x2 + self.blk3(x2)
        x4 = self.output(x3)
        return x4

class FFN(nn.Module):
    def __init__(
            self, 
            n_hidden:int = 128, 
            gaussian_mapping_size:int = 256, 
            gaussian_scale_factor:int = 10
        )->None:
        super().__init__()
        
        #Gausiian Normally Distributed Matrix with mean 0, std 
        self.B = (torch.rand((gaussian_mapping_size, 2)).normal_(mean = 0, std = gaussian_scale_factor) )

        self.blk1 = LinearReLUBlock(gaussian_mapping_size*2, n_hidden//4)
        self.blk2 = LinearReLUBlock(n_hidden//4, n_hidden)
        self.blk3 = LinearReLUBlock(n_hidden, n_hidden)
        self.output = nn.Sequential(
            nn.Linear(n_hidden, 3),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x_proj = (2.*np.pi*x) @ self.B.T
        fourier_terms = [torch.sin(x_proj), torch.cos(x_proj)]
        
        # [[sin(x_coord), cos(x_coord)], [sin(y_coord), cos(y_coord)]]
        new_x = torch.cat(tuple(fourier_terms), dim = -1) 
        
        x1 = self.blk1(new_x)
        x2 = self.blk2(x1)
        x3 = x2 + self.blk3(x2)
        x4 = self.output(x3)
        
        return x4