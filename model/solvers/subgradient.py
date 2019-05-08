import torch
import math

class SubgradientSolver:
    def __init__(self, dim, step_schedule, lamb):
        self.dim = dim
        self.schedule = step_schedule
        self.lamb = lamb

    def optimize(self, obj, constraints):
        x = torch.rand((self.dim, 1))
        # Arbitrarily decided to do 100 iterations - primal dual will fix this.
        for it in range(100):
            g = obj.subgradient_obj(x)
            for constraint in constraints:
                g -= 1 / self.lamb * constraint.subgradeint_cons(x) / constraint.violation(x)
            if type(self.schedule) is float:
                x -= self.schedule * g
            elif self.schedule == "inv":
                x -= 1 / (it + 1) * g
            elif self.schedule == "inv sq root":
                x -= 1 / math.sqrt(it + 1) * g
            else:
                raise Exception("Invalid schedule for subgradient solver.")
        return x
