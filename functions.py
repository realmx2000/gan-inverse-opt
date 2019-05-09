import numpy as np
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, dims, vec=None):
        super().__init__()
        if vec is None:
            self.vec = nn.Parameter(5 * (torch.rand((dims, 1)) - 0.5))
        else:
            self.vec = vec
        self.b = torch.tensor(1)#nn.Parameter(5 * (torch.rand(1) - 0.5))

    def subgradient(self, x):
        return self.vec

    def forward(self, x):
        return torch.matmul(self.vec.t(), x)

    def eval_cp(self, x):
        return x.T @ self.vec.detach().numpy()

    def violation(self, x):
        return torch.matmul(self.vec.t(), x) - self.b

    def violation_cp(self, x):
        return  x.T @ self.vec.detach().numpy() - self.b.detach().numpy()