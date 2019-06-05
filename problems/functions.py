import cvxpy as cp
import torch
import torch.nn as nn
import numpy as np
import warnings
import numpy.linalg as la
from collections import defaultdict

class Linear(nn.Module):
    def __init__(self, dims, vec=None, b=None):
        super().__init__()
        self.dim = dims
        if vec is None:
            self.vec = nn.Parameter(5 * (torch.rand((dims, 1)) - 0.5))
        else:
            self.vec = vec
        if b is None:
            self.b = torch.tensor(1)#nn.Parameter(5 * (torch.rand(1) - 0.5))
        else:
            self.b = b
        self.hess = torch.zeros((dims, dims))
        self.param_list = {}
        if self.vec.requires_grad:
            self.param_list['vec'] = self.vec
        self.param_specs = defaultdict(str)

    def subgradient(self, x):
        return self.vec

    def hessian(self, x):
        return self.hess

    def forward(self, x):
        return torch.matmul(self.vec.t(), x)

    def eval_cp(self, x):
        return x.T @ self.vec.detach().numpy()

    def violation(self, x):
        return torch.matmul(self.vec.t(), x) - self.b

    def violation_cp(self, x):
        return  x.T @ self.vec.detach().numpy() - self.b.detach().numpy()

    #Jacobian w.r.t. parameter of gradient w.r.t. x
    def kkt_hess_partial(self, x):
        D_vec = np.ones((self.dim, self.dim))
        return [D_vec]

    #Jacobian w.r.t parameter of function
    def kkt_grad_partial(self, x):
        x = x.detach().numpy()
        D_vec = x.T
        return [D_vec]

    # dx terms are removed; computed using gradient
    def differential(self, d, x):
        if d['vec'] == 1:
            return [x.t(), 1] #['T', x]
        return [torch.tensor(0)]

    def differential_grad(self, d, x):
        if d['vec'] == 1:
            return ['T']
        return [torch.tensor(0)]

class Quadratic(nn.Module):
    def __init__(self, dims, mat=None, vec=None, b=None):
        super().__init__()
        if mat is None:
            A = 5 * (torch.rand((dims, dims)) - 0.5)
            self.mat = nn.Parameter(A @ A.t())
        else:
            self.mat = mat
        if vec is None:
            self.vec = nn.Parameter(5 * (torch.rand((dims, 1)) - 0.5))#torch.zeros((dims, 1), requires_grad=False)#
        else:
            self.vec = vec
        if b is None:
            self.b = nn.Parameter(5 * (torch.rand(1) - 1.0))
        else:
            self.b = b

        self.param_list = {}
        if self.mat.requires_grad:
            self.param_list['mat'] = self.mat
        if self.vec.requires_grad:
            self.param_list['vec'] = self.vec
        if self.b.requires_grad:
            self.param_list['b'] = self.b
        self.param_specs = defaultdict(str)
        self.param_specs['mat'] = 'psd'

    def subgradient(self, x):
        return self.mat @ x + self.vec

    def hessian(self, x):
        return self.mat

    def forward(self, x):
        self.mat = torch.nn.Parameter(torch.from_numpy(self.nearestPD(self.mat.detach().numpy()).astype(np.float32)))
        return 0.5 * x.t() @ self.mat @ x + self.vec.t() @ x

    def eval_cp(self, x):
        self.mat = torch.nn.Parameter(torch.from_numpy(self.nearestPD(self.mat.detach().numpy()).astype(np.float32)))
        return 0.5 * cp.quad_form(x, self.mat.detach().numpy()) + x.T @ self.vec.detach().numpy()

    def violation(self, x):
        self.mat = torch.nn.Parameter(torch.from_numpy(self.nearestPD(self.mat.detach().numpy()).astype(np.float32)))
        return 0.5 * x.t() @ self.mat @ x + self.vec.t() @ x + self.b

    def violation_cp(self, x):
        self.mat = torch.nn.Parameter(torch.from_numpy(self.nearestPD(self.mat.detach().numpy()).astype(np.float32)))
        return 0.5 * cp.quad_form(x, self.mat.detach().numpy()) + x.T @ self.vec.detach().numpy() + self.b.detach().numpy()

    def kkt_hess_partial(self, x):
        x = x.detach().numpy()
        D_mat = np.zeros((self.dim, self.dim, self.dim))
        for i in range(self.dim):
            D_mat[i, : ,i] = x
        D_vec = np.ones((self.dim, self.dim))
        D_b = np.zeros((1, 1))
        return [D_mat, D_vec, D_b]

    def kkt_grad_partial(self, x):
        x = x.detach().numpy()
        D_mat = x @ x.T
        D_vec = x.T
        D_b = np.ones((1, 1))
        return [D_mat, D_vec, D_b]

    # dx terms are removed
    def differential(self, d, x):
        if d['mat'] == 1:
            return [x, 'T', 0.5 * x.t()] #[0.5 * x.t(), 1, x]
        if d['vec'] == 1:
            return [x.t(), 1] #['T', x]
        if d['b'] == 1:
            return ['T']
        return [torch.tensor(0)]

    def differential_grad(self, d, x):
        if d['mat'] == 1:
            return [x.t(), 'T'] #[1, x]
        if d['vec'] == 1:
            return ['T']
        return [torch.tensor(0)]

class SecondOrderCone(nn.Module):
    def __init__(self, dims, A=None, b=None, c=None, d=None):
        super().__init__()
        if A is None:
            A = 5 * (torch.rand((dims, dims)) - 0.5)
            self.A = nn.Parameter(A @ A.t())
        else:
            self.A = A
        if b is None:
            self.b = nn.Parameter(5 * (torch.rand((dims, 1)) - 0.5))
        else:
            self.b = b
        if c is None:
            self.c = nn.Parameter(5 * (torch.rand((dims, 1)) - 0.5))
        else:
            self.c = c
        if d is None:
            self.d = nn.Parameter(5 * (torch.rand(1) - 0.5))
        else:
            self.d = d

    def subgradient(self, x):
        return 2 * (self.A.t() @ self.A @ x + self.A.t() @ self.b) - self.c

    def hessian(self, x):
        return 2 * self.A.t() @ self.A

    def forward(self, x):
        warnings.warn("Second order cone should only be used as constraint.")
        return self.violation(x)

    def eval_cp(self, x):
        warnings.warn("Second order cone should only be used as constraint.")
        return self.violation_cp(x)

    def violation(self, x):
        return torch.norm(self.A @ x + self.b) - self.c.t() @ x - self.d

    def violation_cp(self, x):
        return cp.norm(self.A.detach().numpy() @ x + self.b.detach().numpy()) - x.T @ self.c.detach().numpy() - self.d.detach().numpy()

class Semidefinite(nn.Module):
    def __init__(self, dims, matrix_dim=None, matrices=None):
        self.dims = dims
        if matrices is None:
            self.matrices = nn.ParameterList()
            for _ in range(dims + 1):
                A = 5 * (torch.rand((matrix_dim, matrix_dim)) - 0.5)
                self.matrices.append(nn.Parameter(A @ A.t()))
        else:
            self.matrices = matrices

        self.param_list = {}
        self.param_specs = defaultdict(str)
        for i in range(dims):
            k = "mat" + str(i)
            self.param_list[k] = self.matrices[i]
            self.param_specs[k] = 'psd'

    def forward(self, x):
        warnings.warn("Second order cone should only be used as constraint.")
        return self.violation(x)

    def eval_cp(self, x):
        warnings.warn("Second order cone should only be used as constraint.")
        return self.violation_cp(x)

    def violation(self, x):
        val = self.matrices[0]
        for i in range(self.dims):
            val = val + x[i] * self.matrices[i + 1]
        return val

    def violation_cp(self, x):
        val = self.matrices[0].detach().numpy()
        for i in range(self.dims):
            val = val + x[i] * self.matrices[i + 1].detach().numpy()
        return val
