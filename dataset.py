import cvxpy as cp
from torch.utils.data import Dataset
import numpy as np
import torch
from problems import generate_LP, generate_QP
from models.solvers import *

class SyntheticDataset(Dataset):
    def __init__(self, prob_type, num_constraints, dim, mat=None, vec=None):
        super().__init__()
        self.dim = dim
        if prob_type == "LP":
            self.problem = generate_LP(dim, num_constraints, vec=vec)
        elif prob_type == "QP":
            self.problem = generate_QP(dim, num_constraints, mat=mat, vec=vec)
        else:
            raise Exception("Objective type %s invalid." % prob_type)

        self.data = self.get_solutions()
        self.length = len(self.data)

    def __len__(self):
        return 1 #Chosen arbitrarily

    def __getitem__(self, index):
        # Balanced train set
        idx = index % self.length
        item = self.data[idx]
        return item

    def get_solutions(self):
        x, val = self.problem.solve_cp()
        assert(x is not None) # Just rerun until this passes

        minimizers = [x]
        solver = NewtonSolver(self.dim, "inv sq root", 1000.0)
        for _ in range(len(self)):
            #w = np.random.randn(self.dim, 1)
            x = cp.Variable((self.dim, 1))
            #obj = cp.Minimize(w.T @ x)
            obj = cp.Minimize(self.problem.obj.eval_cp(x))
            #constraints = [self.problem.eval_cp(x) <= val]
            constraints = []
            for constraint in self.problem.constraints:
                constraints.append(constraint.violation_cp(x) <= 0)
            prob = cp.Problem(obj, constraints)
            prob.solve(solver="ECOS")
            if any([(np.abs(x.value - y) > 1e-4).all() for y in minimizers]):
                minimizers.append(x.value)

        minimizers = [torch.from_numpy(x[:, 0]).float() for x in minimizers]

        return minimizers
