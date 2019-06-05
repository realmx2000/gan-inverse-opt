import torch
import cvxpy as cp
import numpy as np
from collections import defaultdict

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

    def KKT_hessian(self, x, lamb):
        x = x.t()
        Df = np.zeros((len(self.constraints), self.dim))
        f = np.zeros(len(self.constraints))
        hess = self.obj.hessian(x).detach().numpy()
        for i, constraint in enumerate(self.constraints):
            hess += lamb[i] * constraint.hessian(x).detach().numpy()
            Df[i,:] = constraint.subgradient(x).detach().numpy().T
            f[i] = constraint.violation(x).detach().numpy()

        kkt_hess = np.block([[hess, Df.T], [np.diag(lamb.squeeze()) @ Df, np.diag(f)]])
        return kkt_hess

    def calc_gradients(self, x, lamb):
        lamb_vec = lamb.squeeze(1)
        kkt_hess = self.KKT_hessian(x, lamb).astype(np.float32)
        grad = x.grad
        x = x.detach().numpy()
        full_grad = torch.cat([grad, torch.zeros((1, len(self.constraints)))], dim=1)
        d = np.linalg.solve(kkt_hess, full_grad.numpy().T)
        dx = d[:x.shape[1]]
        dlamb = d[x.shape[1]:]

        #dh = np.diag(lamb_vec) @ dlamb
        dG = -np.diag(lamb_vec) @ (dlamb @ x + lamb @ dx.T)
        dQ = -0.5 * (dx @ x + x.T @ dx.T)
        dq = -dx
        #self.obj.mat.grad = torch.from_numpy(dQ)
        #self.obj.vec.grad = torch.from_numpy(dq)
        for i, constraint in enumerate(self.constraints):
            constraint.vec.grad = torch.from_numpy(np.expand_dims(dG[i, :], 1))
            #constraint.b.grad = torch.from_numpy(dh[i])

    def multiply(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        if x.numel() == 1 or y.numel() == 1:
            return x * y
        else:
            return x @ y

    def eval_term(self, term):
        post = None
        pre = None
        found = False
        for i, item in enumerate(term):
            if isinstance(item, int) and item == 1:
                transpose = True
                found = True
                continue
            elif isinstance(item, str) and item == 'T':
                transpose = False
                found = True
                continue
            if not found:
                post = item if post is None else self.multiply(post, item)
            else:
                pre = item if pre is None else self.multiply(pre, item)
        if not found:
            return 0
        return self.multiply(pre, post) if not transpose else self.multiply(post, pre).t()


    def calc_gradients_general(self, x, lamb):
        kkt_hess = self.KKT_hessian(x, lamb)
        lamb = torch.from_numpy(lamb)
        grad = x.grad
        x = x.t()
        full_grad = torch.cat([grad, torch.zeros((1, len(self.constraints)))], dim=1)
        d = -np.linalg.solve(kkt_hess, full_grad.numpy().T).astype(np.float32)
        dx = torch.from_numpy(d[:x.shape[0]])
        dlamb = torch.from_numpy(d[x.shape[0]:])
        if len(self.obj.param_list) > 0:
            perturb = defaultdict(int)
            for name, param in self.obj.param_list.items():
                perturb[name] = 1
                term = self.obj.differential_grad(perturb, x)
                term.append(dx)
                gradient = self.eval_term(term)
                param.grad = gradient
                perturb[name] = 0

        for i, constraint in enumerate(self.constraints):
            perturb = defaultdict(int)
            for name, param in constraint.param_list.items():
                perturb[name] = 1
                term1 = constraint.differential_grad(perturb, x) + [lamb[i] * lamb[i]] #TODO: Unsure why D(lambda) is multiplied in
                term1.append(dx)
                grad_1 = self.eval_term(term1)
                term2 = constraint.differential(perturb, x) + [lamb[i]]
                term2.append(dlamb[i])
                grad_2 = self.eval_term(term2)
                gradient = grad_1 + grad_2
                if constraint.param_specs[name] == "psd":
                    gradient = 0.5 * (gradient + gradient.t())
                param.grad = gradient
                perturb[name] = 0
