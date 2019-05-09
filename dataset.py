import cvxpy as cp
from torch.utils.data import Dataset
import numpy as np
from functions import *

class SyntheticDataset(Dataset):
    def __init__(self, obj_type, constraint_type, num_constraints, dim, obj=None):
        super.__init__()
        self.dim = dim
        if obj_type == "linear":
            self.obj = Linear(dim, obj)
        else:
            raise Exception("Objective type %s invalid." % obj_type)

        self.constraints = []
        if constraint_type == "linear":
            for _ in range(num_constraints):
                self.constraints.append(Linear(dim))
        else:
            raise Exception("Constraint type %s invalid." % constraint_type)

        self.data = self.get_solutions()
        if len(self.data) == 0:
            self.__init__(obj_type, constraint_type, num_constraints, dim)
        self.length = len(self.data)

    def __len__(self):
        return 1000 #Chosen arbitrarily

    def __getitem__(self, index):
        # Balanced train set
        idx = index % self.length
        item = self.data[idx]
        return item

    def get_solutions(self):
        x = cp.Variable((self.dim, 1))
        obj = cp.Minimize(self.obj.eval_cp)
        constraints = []
        for constraint in self.constraints:
            constraints.append(constraint.violation_cp(x) <= 0)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        #TODO: Find a way to get all optimal points
        if prob.status != "optimal":
            return []
        else:
            return x.value
