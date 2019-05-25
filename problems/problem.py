import torch
import cvxpy as cp
import numpy as np

class Problem(torch.nn.Module):
    def __init__(self, dim, obj, constraints, lamb=1000.0):
        super().__init__()
        self.dim = dim
        self.obj = obj
        self.constraints = constraints
        self.lamb = lamb

    def forward(self, x):
        return self.obj(x)

    def eval_cp(self, x):
        return self.obj.eval_cp(x)

    def eval_barrier(self, x):
        val = self.obj(x)
        for constraint in self.constraints:
            val -= torch.log(-constraint.violation(x)) / self.lamb
        if torch.isnan(val):
            val = torch.tensor(float('inf'))
        return val

    def solve(self, solver):
        return solver.optimize(self)

    def solve_cp(self):
        x = cp.Variable((self.dim, 1))
        obj = cp.Minimize(self.obj.eval_cp(x))
        constraints = []
        for constraint in self.constraints:
            constraints.append(constraint.violation_cp(x) <= 0)
        prob = cp.Problem(obj, constraints)
        prob.solve(solver="ECOS")
        dual = [constraint.dual_value.squeeze() for constraint in constraints]
        dual = np.expand_dims(np.asarray(dual), 1)

        return x.value.astype(np.float32), prob.value.astype(np.float32), dual.astype(np.float32)

    def derivatives(self, x):
        hess = self.obj.hessian(x)
        g = self.obj.subgradient(x)
        for constraint in self.constraints:
            grad = constraint.subgradient(x)
            g = g - (1 / self.lamb) * (grad / constraint.violation(x))

            barrier_hess = torch.matmul(grad, grad.t()) / (constraint.violation(x) ** 2)
            barrier_hess = barrier_hess - constraint.hessian(x) / constraint.violation(x)
            hess = hess + barrier_hess / self.lamb
        return g, hess

    def phase_1(self):
        if len(self.constraints) > 0:
            x = cp.Variable((self.dim, 1))
            obj = cp.Minimize(0)
            constraints_cp = []
            for constraint in self.constraints:
                constraints_cp.append(constraint.violation_cp(x) <= -0.01)
            prob = cp.Problem(obj, constraints_cp)
            prob.solve()
        else:
            return np.random.randn(self.dim, 1)
        return x.value