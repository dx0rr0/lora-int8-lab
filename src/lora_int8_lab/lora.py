"""
Lora consists of learning a lower rank approximation of the weight updates using two smaller matrices (A and B)
"""


import torch.nn as nn
import torch

class LoRALayer(nn.Module):
    def __init__(self, rank: int, layer: nn.Module, alpha: int = 16) -> None:
        super().__init__()
        self.layer = layer.requires_grad_(False)
        self._in_features = layer.weight.shape[0]
        self._out_features = layer.weight.shape[1]
        if isinstance(layer, nn.Linear):     
            self._in_features = layer.weight.shape[1]
            self._out_features = layer.weight.shape[0]
        self.B = nn.Parameter(torch.zeros(self._out_features, rank))
        self.A = nn.Parameter(0.01*torch.randn(rank, self._in_features))
        self.scaling = alpha/rank
    
    
    def forward(self, x):
        return self.layer(x) + self.scaling * (x @ self.A.T @ self.B.T)
