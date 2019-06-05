from models import *
from dataset import SyntheticDataset
from arg_parser import ArgParser
from torch.utils.data import DataLoader
import torch.optim as optim
from models.solvers import NewtonSolver
import torch
import numpy as np

def get_optimizer(model, opt, lr, momentum, reg):
    if opt == "nesterov":
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg, nesterov=True)
    elif opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg)
    elif opt == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    else:
        raise Exception("Invalid optimizer: %s", opt)

    return optimizer

def get_scheduler(optimizer, scheduler, decay_step, decay_rate, restart, patience):
    if scheduler == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=decay_step, gamma=decay_rate)
    elif scheduler == 'cos':
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=restart)
    elif scheduler == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=decay_rate, patience=patience)
    else:
        return None

def verify_solution(dim, objects):
    vals = []
    for obj in objects:
        x1, _ = obj.problem.solve_cp()
        x2 = obj.problem.solve(NewtonSolver(dim, "inv sq root", 1000.0))
        vals.append(x1)
        vals.append(x2)
    print("Verification: ")
    print(np.linalg.norm(vals[0] - vals[2]) < 1e-3)

"""
def analytic_gradients(prob, generated):
    Q = prob.obj.mat.detach().numpy()
    G = np.zeros((len(prob.constraints), prob.dim)).astype(np.float32)
    h = np.zeros((len(prob.constraints), 1)).astype(np.float32)
    for i, constraint in enumerate(prob.constraints):
        G[i,:] = constraint.vec.detach().numpy().squeeze()
        h[i,:] = constraint.b.detach().numpy()
    x, _, lamb = prob.solve_cp()
    lamb_vec = lamb.squeeze(1)
    diff = np.block([[Q, G.T @ np.diag(lamb_vec)], [G, np.diag((G @ x - h).squeeze(1))]])
    grad = np.concatenate([generated.grad.numpy().T, np.zeros((len(prob.constraints), 1)).astype(np.float32)])
    d = np.linalg.solve(diff, grad)
    dx = d[:x.shape[0]]
    dlamb = d[x.shape[0]:]

    dh = np.diag(lamb_vec) @ dlamb
    dG = -np.diag(lamb_vec) @ (dlamb @ x.T + lamb @ dx.T)
    for i, constraint in enumerate(prob.constraints):
        constraint.vec.grad = torch.from_numpy(np.expand_dims(dG[i,:], 1))
        constraint.b.grad = torch.from_numpy(dh[i])
"""

#TODO: Discriminator and Generator both train to near perfect performance independently, but the GAN doesn't converge.
#Looks like generated vectors often blow up after a couple epochs.
#TODO: Generalize the KKT gradients

def train(args):
    #torch.manual_seed(6)
    #np.random.seed(6)
    #mat = 5 * (torch.rand((args.dim, args.dim), requires_grad=False) - 0.5)
    #mat = mat @ mat.t()
    #vec = 5 * (torch.rand((args.dim, 1), requires_grad=False) - 0.5)

    generator = Generator(args.prob_type, args.num_constraints, args.dim, args.solver, args.solve_schedule,
                          args.lamb, args.constraint_dims) #mat=mat, vec=vec,
    #discriminator = Discriminator(args.dim, args.activation)

    dataset = SyntheticDataset(args.prob_type, args.num_constraints, args.dim, args.constraint_dims) #, mat=mat, vec=vec,
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    opt_gen = get_optimizer(generator, args.optimizer, args.lr_g, args.momentum, args.reg)
    #opt_discrim = get_optimizer(discriminator, args.optimizer, args.lr_d, args.momentum, args.reg)

    sched_gen = get_scheduler(opt_gen, args.gen_scheduler, args.gen_decay_step, args.gen_decay_rate,
                              args.gen_restart, args.gen_patience)

    label_true = torch.tensor([1], dtype=torch.float)
    label_fake = torch.tensor([0], dtype=torch.float)

    #loss_fn = lambda x, y: torch.norm(x - y, p=float('inf'))
    loss_fn = torch.nn.L1Loss(reduction='sum')
    #TODO: Currently batch size is 1. Does it make sense to increase?
    for epoch in range(args.epochs):
        losses_D = []
        losses_G = []
        for minimizer in dataloader:
            """
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
            """

            generator.zero_grad()
            generated, dual = generator()
            generated = generated.t()
            generated.retain_grad()
            loss_G = loss_fn(generated, minimizer)
            loss_G = loss_G
            #pred_gen, loss_G = discriminator(generated, label_true)
            loss_G.backward()
            #analytic_gradients(generator.problem, generated)
            generator.backward(generated, dual)
            #torch.nn.utils.clip_grad_value_(generator.parameters(), 10)
            opt_gen.step()
            losses_G.append(loss_G.detach().item())

        if sched_gen is not None:
            sched_gen.step(np.mean(losses_G))
        print(minimizer)
        print(generated)
        """
        print(dataset.problem.constraints[0].vec)
        print(generator.problem.constraints[0].vec)
        print(dataset.problem.constraints[0].b)
        print(generator.problem.constraints[0].b)
        """

        print("Epoch %d, Generator loss %f, Discriminator loss %f" % (epoch, np.mean(losses_G), np.mean(losses_D)))
    verify_solution(args.dim, [generator, dataset])


if __name__ == "__main__":
    parser = ArgParser()
    args = parser.parse_args()
    train(args)