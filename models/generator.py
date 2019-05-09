import numpy as np
from functions import *
from .solvers import *
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, obj_type, constraint_type, num_constraints, dim, solver_type, schedule, lamb, obj=None):
        super().__init__()
        if obj_type == "linear":
            self.obj = Linear(dim, obj)
        else:
            raise Exception("Objective type %s invalid." % obj_type)

        self.constraints = nn.ModuleList()
        if constraint_type == "linear":
            for _ in range(num_constraints):
                self.constraints.append(Linear(dim))
        else:
            raise Exception("Constraint type %s invalid." % constraint_type)

        if solver_type == "subgradient":
            self.solver = SubgradientSolver(dim, schedule, lamb)
        else:
            raise Exception("Solver type %s invalid." % solver_type)

    def forward(self):
        minimizer = self.solver.optimize(self.obj, self.constraints)
        return minimizer
