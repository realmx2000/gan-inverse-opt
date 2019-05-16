from models import *
from dataset import SyntheticDataset
from arg_parser import ArgParser
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
from models.solvers import NewtonSolver
import torch
import numpy as np

def get_optimizer(model, opt, lr, momentum, reg):
    if opt == "nesterov":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg, nesterov=True)
    elif opt == "sgd":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg)
    elif opt == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=reg)
    else:
        raise Exception("Invalid optimizer: %s", opt)

    return optimizer

def verify_solution(dim, objects):
    vals = []
    for obj in objects:
        x1, _ = obj.problem.solve_cp()
        x2 = obj.problem.solve(NewtonSolver(dim, "inv sq root", 1000.0))
        vals.append(x1)
        vals.append(x2)
    print("Verification: ")
    print(np.linalg.norm(vals[0] - vals[2]) < 1e-3)


#TODO: Gradient updates sometimes make problem infeasible, then phase 1 fails.
#TODO: Hessians are sometimes singular - does this have to do with improper gradient updates? Doesn't seem like it should be.
#TODO: Discriminator and Generator both train to near perfect performance independently, but the GAN doesn't converge.
#Looks like generated vectors often blow up after a couple epochs.

def train(args):
    vec = 5 * (torch.rand((args.dim, 1), requires_grad=False) - 0.5)

    generator = Generator(args.prob_type, args.num_constraints,
                          args.dim, args.solver, args.solve_schedule, args.lamb, obj=vec)
    discriminator = Discriminator(args.dim, args.activation)

    dataset = SyntheticDataset(args.prob_type, args.num_constraints, args.dim, obj=vec)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    opt_gen = get_optimizer(generator, args.optimizer, args.lr_g, args.momentum, args.reg)
    opt_discrim = get_optimizer(discriminator, args.optimizer, args.lr_d, args.momentum, args.reg)

    label_true = torch.tensor([1], dtype=torch.float)
    label_fake = torch.tensor([0], dtype=torch.float)

    #loss_fn = torch.nn.MSELoss()
    #TODO: Currently batch size is 1. Does it make sense to increase?
    for epoch in range(args.epochs):
        losses_D = []
        losses_G = []
        for minimizer in dataloader:
            for _ in range(args.train_ratio):
                discriminator.zero_grad()
                generated = generator().t()
                #generated = torch.randn((1, args.dim))
                pred_real, loss_real = discriminator(minimizer, label_true)
                pred_gen, loss_gen = discriminator(generated.detach(), label_fake)

                loss_D = loss_real + loss_gen
                loss_D.backward()
                opt_discrim.step()
                losses_D.append(loss_D.detach().item())


            generator.zero_grad()
            generated = generator().t()
            #loss_G = loss_fn(generated, minimizer)

            pred_gen, loss_G = discriminator(generated, label_true)
            loss_G.backward()
            torch.nn.utils.clip_grad_value_(generator.parameters(), 10)
            opt_gen.step()
            losses_G.append(loss_G.detach().item())

        print(minimizer)
        print(generated)

        print("Epoch %d, Generator loss %f, Discriminator loss %f" % (epoch, np.mean(losses_G), np.mean(losses_D)))
    verify_solution(args.dim, [generator, dataset])


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    train(args)