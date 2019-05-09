import cvxpy as cp
from torch.utils.data import Dataset
from models.solvers.subgradient import SubgradientSolver
from functions import *

class SyntheticDataset(Dataset):
    def __init__(self, obj_type, constraint_type, num_constraints, dim, obj=None):
        super().__init__()
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

        for constraint in self.constraints:
            print(constraint.vec)
            print(constraint.b)

    def __len__(self):
        return 10 #Chosen arbitrarily

    def __getitem__(self, index):
        # Balanced train set
        idx = index % self.length
        item = self.data[idx]
        return item

    def get_solutions(self):
        x = cp.Variable((self.dim, 1))
        obj = cp.Minimize(self.obj.eval_cp(x))
        constraints = []
        for constraint in self.constraints:
            constraints.append(constraint.violation_cp(x) <= 0)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print(prob.status)

        """
        solver = SubgradientSolver(self.dim, "inv sq root", 100.0)
        x_test = solver.optimize(self.obj, self.constraints)
        print(x.value)
        print(x_test)
        input()
        """

        #TODO: Find a way to get all optimal points
        if prob.status != "optimal":
            return []
        else:
            return [torch.from_numpy(x.value[:,0]).float()]
