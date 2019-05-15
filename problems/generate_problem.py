import numpy as np
import torch
from problems import *

def generate_LP(dim, num_constraints, lamb=1000.0, vec=None):
    A = torch.randn((num_constraints, dim))
    x = torch.randn((dim, 1))
    b = A @ x + torch.rand((num_constraints, 1))
    obj = Linear(dim, vec=vec)
    constraints = torch.nn.ModuleList()
    for i in range(num_constraints):
        constraints.append(Linear(dim, torch.nn.Parameter(A[i,:].unsqueeze(1)), b[i]))
    return Problem(dim, obj, constraints, lamb)
