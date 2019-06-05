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
    obj = Quadratic(dim, mat, vec, torch.tensor(0, requires_grad=False))
    if num_constraints > 0:
        A = torch.randn((num_constraints, dim))
        x = torch.randn((dim, 1))
        b = A @ x + torch.rand((num_constraints, 1))
        constraints = torch.nn.ModuleList()
        for i in range(num_constraints):
            constraints.append(Linear(dim, torch.nn.Parameter(A[i,:].unsqueeze(1)), torch.nn.Parameter(b[i])))
    else:
        constraints = []
    return Problem(dim, obj, constraints, lamb)

def test_QCQP(dim):
    A = torch.randn((dim, dim))
    b = torch.randn((dim, 1))
    obj = Quadratic(dim, mat= A.t() @ A, vec= A.t() @ b, b=b.t() @ b)
    cons_mat = torch.randn((dim, dim)) #torch.eye(dim)#
    cons_mat = cons_mat @ cons_mat.t()
    vec = torch.randn((dim, 1)) #torch.zeros((dim, 1)) #
    cons = -torch.rand((1, 1)) #torch.tensor(-1)#
    constraint = [Quadratic(dim, mat=cons_mat, vec=vec, b=cons)]
    return Problem(dim, obj, constraint, 1000.0)

def generate_QCQP(dim, num_constraints, lamb=1000.0, mat=None, vec=None):
    obj = Quadratic(dim, mat, vec, torch.tensor(0, requires_grad=False))

    #TODO: How to make this feasible?
    if num_constraints > 0:
        constraints = torch.nn.ModuleList()
        for i in range(num_constraints):
            vec = torch.zeros((dim, 1), requires_grad=False)
            #cons = torch.tensor((-1), requires_grad=False)
            constraints.append(Quadratic(dim, vec=vec))
    else:
        constraints = []
    return Problem(dim, obj, constraints, lamb)

def generate_SOCP(dim, constraint_dims, lamb=1000.0, vec=None):
    obj = Linear(dim, vec=vec)
    if len(constraint_dims) > 0:
        constraints = torch.nn.ModuleList()
        x0 = torch.randn((dim, 1))
        for constraint_dim in constraint_dims:
            A = torch.nn.Parameter(torch.randn((constraint_dim, dim)))
            b = torch.nn.Parameter(torch.randn((constraint_dim, 1)))
            c = torch.nn.Parameter(torch.randn((dim, 1)))
            d = torch.nn.Parameter(torch.norm(A @ x0 +b) - x0.t() @ c) #+ torch.rand(1) TODO: Why?
            constraints.append(SecondOrderCone(dim, A, b, c, d))
    else:
        constraints = []
    return Problem(dim, obj, constraints, lamb)

def generate_SDP(dim, num_constraints, matrix_dim, lamb=1000.0, vec=None):
    obj = Linear(dim, vec=vec)
    constraints = torch.nn.ModuleList()
    constraints.append(Semidefinite(dim, matrix_dim))
    return Problem(dim, obj, constraints, lamb)
