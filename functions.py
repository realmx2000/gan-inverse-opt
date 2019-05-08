import numpy as np
import torch
import cvxpy as cp

class Linear:
    def __init__(self, dims):
        self.vec = torch.rand((dims, 1))
        self.b = torch.rand(1)

    def set_params(self, a):
        self.vec = a

    def get_params(self):
        return self.vec

    def subgradient_obj(self, x):
        return self.vec

    def subgradient_cons(self, x):
        return -self.vec

    def eval(self, x):
        return torch.dot(x, self.vec)

    def eval_cp(self, x):
        return x.T @ self.vec.numpy()

    def violation(self, x):
        return self.b - torch.dot(x, self.vec)

    def violation_cp(self, x):
        return self.b - x.T @ self.vec.numpy()