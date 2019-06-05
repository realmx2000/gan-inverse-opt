from problems import *
from .solvers import *
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, prob_type, num_constraints, dim, solver_type, schedule, lamb, constraint_dims=None, mat=None, vec=None):
        super().__init__()

        if prob_type == "LP":
            self.problem = generate_LP(dim, num_constraints, lamb=lamb, vec=vec)
        elif prob_type == "QP":
            self.problem = generate_QP(dim, num_constraints, lamb=lamb, mat=mat, vec=vec)
        elif prob_type == "QCQP":
            self.problem = generate_QCQP(dim, num_constraints, lamb=lamb, mat=mat, vec=vec)
        elif prob_type == "SOCP":
            self.problem = generate_SOCP(dim, constraint_dims, vec=vec)
        else:
            raise Exception("Problem type %s invalid." % prob_type)


        if solver_type == "subgradient":
            self.solver = SubgradientSolver(dim, schedule, lamb)
        elif solver_type == "newton":
            self.solver = NewtonSolver(dim, schedule, lamb)
        else:
            raise Exception("Solver type %s invalid." % solver_type)

    def forward(self):
        #minimizer = self.problem.solve(self.solver)
        minimizer, _, dual = self.problem.solve_cp()
        return torch.tensor(minimizer.astype(np.float32), requires_grad=True), dual

    def backward(self, x, lamb):
        #self.problem.calc_gradients(x, lamb)
        self.problem.calc_gradients_general(x, lamb)
