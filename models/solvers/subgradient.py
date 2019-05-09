import torch
import math
import numpy as np

class SubgradientSolver:
    def __init__(self, dim, step_schedule, lamb):
        self.dim = dim
        self.schedule = step_schedule
        self.lamb = lamb

    def optimize(self, obj, constraints):
        x = torch.zeros((self.dim, 1))

        for it in range(500):
            g = obj.subgradient(x)
            for constraint in constraints:
                if constraint.violation(x) >= 0.01:
                    g = constraint.subgradient(x)
                    break
                else:
                    g = g - (1 / self.lamb) * (constraint.subgradient(x) / constraint.violation(x))

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

