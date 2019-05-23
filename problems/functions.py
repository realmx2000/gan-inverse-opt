import cvxpy as cp
import torch
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, dims, vec=None, b=None):
        super().__init__()
        if vec is None:
            self.vec = nn.Parameter(5 * (torch.rand((dims, 1)) - 0.5))
        else:
            self.vec = vec
        if b is None:
            self.b = torch.tensor(1)#nn.Parameter(5 * (torch.rand(1) - 0.5))
        else:
            self.b = b
        self.hess = torch.zeros((dims, dims))

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

class Quadratic(nn.Module):
    def __init__(self, dims, mat=None, vec=None, b=None):
        super().__init__()
        if mat is None:
            A = 5 * (torch.rand((dims, dims)) - 0.5)
            self.mat = nn.Parameter(A @ A.t())
        else:
            self.mat = mat
        if vec is None:
            self.vec = nn.Parameter(5 * (torch.rand((dims, 1)) - 0.5))
        else:
            self.vec = vec
        if b is None:
            self.b = nn.Parameter(5 * (torch.rand(1) - 0.5))
        else:
            self.b = b
        self.hess = self.mat

    def subgradient(self, x):
        return self.hess @ x + self.vec

    def hessian(self, x):
        return self.hess

    def forward(self, x):
        return 0.5 * x.t() @ self.mat @ x + self.vec.t() @ x

    def eval_cp(self, x):
        return 0.5 * cp.quad_form(x, self.mat.detach().numpy()) + x.T @ self.vec.detach().numpy()

    def violation(self, x):
        return 0.5 * x.t() @ self.mat @ x + self.vec.t() @ x - self.b

    def violation_cp(self, x):
        return 0.5 * cp.quad_form(x, self.mat.detach().numpy()) + x.T @ self.vec.detach().numpy() - self.b.detach().numpy()