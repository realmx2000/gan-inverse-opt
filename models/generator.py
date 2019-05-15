import numpy as np
from problems import Problem, generate_LP
from .solvers import *
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, prob_type, num_constraints, dim, solver_type, schedule, lamb, obj=None):
        super().__init__()

        if prob_type == "LP":
            self.problem = generate_LP(dim, num_constraints, lamb=lamb, vec=obj)
        else:
            raise Exception("Problem type %s invalid." % prob_type)


        if solver_type == "subgradient":
            self.solver = SubgradientSolver(dim, schedule, lamb)
        elif solver_type == "newton":
            self.solver = NewtonSolver(dim, schedule, lamb)
        else:
            raise Exception("Solver type %s invalid." % solver_type)

    def forward(self):
        minimizer = self.problem.solve(self.solver)
        return minimizer
