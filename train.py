from models import *
from dataset import SyntheticDataset
from arg_parser import ArgParser
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import cvxpy as cp
import torch
import numpy as np

def get_optimizer(model, opt, lr, momentum, reg):
    if opt == "nesterov":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg, nesterov=True)
    elif opt == "sgd":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg)
    elif opt == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=reg)

    return optimizer

def verify_solution(dim, problems):
    vals = []
    for problem in problems:
        x = cp.Variable((dim, 1))
        obj = cp.Minimize(problem.obj.eval_cp(x))
        constraints = []
        for constraint in problem.constraints:
            constraints.append(constraint.violation_cp(x) <= 0)
        prob = cp.Problem(obj, constraints)
        prob.solve()
        print(prob.status)
        vals.append(x.value)
    print(vals[0])
    print(vals[1])
    print(np.linalg.norm(vals[0] - vals[1]))

def train(args):
    vec = 5 * (torch.rand((args.dim, 1), requires_grad=False) - 0.5)

    generator = Generator(args.obj_type, args.constraint_type, args.num_constraints,
                          args.dim, args.solver, args.solve_schedule, args.lamb, obj=vec)
    discriminator = Discriminator(args.dim, args.activation)

    dataset = SyntheticDataset(args.obj_type, args.constraint_type, args.num_constraints, args.dim, obj=vec)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    opt_gen = get_optimizer(generator, args.optimizer, args.lr, args.momentum, args.reg)
    opt_discrim = get_optimizer(discriminator, args.optimizer, args.lr, args.momentum, args.reg)

    label_true = torch.tensor([1], dtype=torch.float)
    label_fake = torch.tensor([0], dtype=torch.float)

    #for i in range(3):
    #    generator.constraints[i].vec = dataset.constraints[i].vec
    loss_fn = torch.nn.MSELoss()
    #TODO: Currently batch size is 1. Does it make sense to increase?
    for epoch in range(args.epochs):
        losses_D = []
        losses_G = []
        for minimizer in dataloader:
            """
            discriminator.zero_grad()
            generated = generator().t()
            pred_real, loss_real = discriminator(minimizer, label_true)
            pred_gen, loss_gen = discriminator(generated.detach(), label_fake)

            loss_D = loss_real + loss_gen
            loss_D.backward()
            opt_discrim.step()
            losses_D.append(loss_D.detach().item())
            """
            #for _ in range(args.train_ratio):
            generator.zero_grad()
            generated = generator().t()
            loss_G = loss_fn(generated, minimizer)

            #pred_gen, loss_G = discriminator(generated, label_true)
            loss_G.backward()
            opt_gen.step()
            losses_G.append(loss_G.detach().item())

        print("Epoch %d, Generator loss %f" % (epoch, np.mean(losses_G) ))
        for constraint in generator.constraints:
            print(constraint.vec)
            print(constraint.b)
        print(minimizer)
        print(generated)
    verify_solution(args.dim, [generator, dataset])


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    train(args)