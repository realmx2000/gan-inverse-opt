import numpy as np
import torch
from problems import *

def generate_LP(dim, num_constraints, lamb=1000.0, vec=None):
    obj = Linear(dim, vec=vec)
    if num_constraints > 0:
        A = torch.randn((num_constraints, dim))
        x = torch.randn((dim, 1))
        b = A @ x + torch.rand((num_constraints, 1))
        constraints = torch.nn.ModuleList()
        for i in range(num_constraints):
            constraints.append(Linear(dim, torch.nn.Parameter(A[i,:].unsqueeze(1)), b[i]))
    else:
        constraints = []
    return Problem(dim, obj, constraints, lamb)

def generate_QP(dim, num_constraints, lamb=1000.0, mat=None, vec=None):
    obj = Quadratic(dim, mat, vec, 0)
    if num_constraints > 0:
        A = torch.randn((num_constraints, dim))
        x = torch.randn((dim, 1))
        b = A @ x + torch.rand((num_constraints, 1))
        constraints = torch.nn.ModuleList()
        for i in range(num_constraints):
            constraints.append(Linear(dim, torch.nn.Parameter(A[i,:].unsqueeze(1)), b[i]))
    else:
        constraints = []
    return Problem(dim, obj, constraints, lamb)
