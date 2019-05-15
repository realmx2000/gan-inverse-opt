import cvxpy as cp
from torch.utils.data import Dataset
import numpy as np
import torch
from problems import generate_LP, Problem
from models.solvers import *

class SyntheticDataset(Dataset):
    def __init__(self, prob_type, num_constraints, dim, obj=None):
        super().__init__()
        self.dim = dim
        if prob_type == "LP":
            self.problem = generate_LP(dim, num_constraints, vec=obj)
        else:
            raise Exception("Objective type %s invalid." % prob_type)

        self.data = self.get_solutions()
        self.length = len(self.data)

    def __len__(self):
        return 100 #Chosen arbitrarily

    def __getitem__(self, index):
        # Balanced train set
        idx = index % self.length
        item = self.data[idx]
        return item

    def get_solutions(self):
        x, val = self.problem.solve_cp()
        assert(x is not None) # Just rerun until this passes

        minimizers = [x]
        for _ in range(len(self)):
            w = np.random.randn(self.dim, 1)
            x = cp.Variable((self.dim, 1))
            obj = cp.Minimize(w.T @ x)
            constraints = [self.problem.eval_cp(x) <= val]
            for constraint in self.problem.constraints:
                constraints.append(constraint.violation_cp(x) <= 0)
            prob = cp.Problem(obj, constraints)
            prob.solve()
            if any([(np.abs(x.value, y) <= 1e-4).all() for y in minimizers]):
                minimizers.append(x.value)

        minimizers = [torch.from_numpy(x[:, 0]).float() for x in minimizers]

        return minimizers
