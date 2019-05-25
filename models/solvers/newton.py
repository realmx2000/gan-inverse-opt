import torch
import math
import numpy as np

class NewtonSolver:
    def __init__(self, dim, step_schedule, lamb):
        self.dim = dim
        self.step_schedule = step_schedule,
        self.lamb = lamb

    def backtrack_search(self, problem, x, x_step, g):
        alpha = 0.25
        beta = 0.5
        t = 1
        curr = problem.eval_barrier(x)
        while problem.eval_barrier(x + t * x_step) > curr + alpha * t * g.t() @ x_step:
            t = beta * t
        return x + t * x_step

    def optimize(self, problem):
        x = torch.from_numpy(problem.phase_1().astype(np.float32))

        for it in range(500):
            g, hess = problem.derivatives(x)
            hessinv = torch.inverse(hess)
            step = -hessinv @ g
            decrement = g.t() @ hessinv @ g

            x = self.backtrack_search(problem, x, step, g)
            if decrement / 2 < 0.0001:
                break
        #print(it)
        return x