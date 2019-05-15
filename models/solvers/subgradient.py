import torch
import math
import numpy as np

class SubgradientSolver:
    def __init__(self, dim, step_schedule, lamb):
        self.dim = dim
        self.schedule = step_schedule
        self.lamb = lamb

    def optimize(self, problem):
        x = torch.zeros((self.dim, 1))

        for it in range(500):
            g, _ = problem.derivatives(x)
            for constraint in problem.constraints:
                if constraint.violation(x) >= 0.01:
                    g = constraint.subgradient(x)
                    break

            if type(self.schedule) is float:
                x = x - self.schedule * g
            elif self.schedule == "inv":
                x = x - 1 / (it + 1) * g
            elif self.schedule == "inv sq root":
                x = x - 1 / math.sqrt(it + 1) * g
            else:
                raise Exception("Invalid schedule for subgradient solver.")
            if it >= 100 and np.random.rand() < 0.005:
                break

        return x

