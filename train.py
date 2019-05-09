from .models import *
from dataset import SyntheticDataset
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam
import torch.nn as nn
import torch
import numpy as np

def get_optimizer(model, opt, lr, momentum, reg):
    if opt == "nesterov":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg, nesterov=True)
    elif opt == "sgd":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg)
    elif opt == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=reg)

def train(args):
    vec = torch.rand((args.dim, 1), requires_grad=False)

    generator = Generator(args.obj_type, args.constraint_type, args.num_constraints,
                          args.dim, args.solver, args.solve_schedule, args.lamb, obj=vec)
    discriminator = Discriminator(args.dim, args.activation)

    dataset = SyntheticDataset(args.obj_type, args.constraint_type, args.num_constraints, args.dim, obj=vec)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt_gen = get_optimizer(generator, args.optimizer, args.lr, args.momentum, args.reg)
    opt_discrim = get_optimizer(discriminator, args.optimizer, args.lr, args.momentum, args.reg)

    loss_fn = nn.BCELoss()
    #TODO: Currently batch size is 1. Does it make sense to increase?
    for epoch in range(args.epochs):
        losses_D = []
        losses_G = []
        for minimizer in dataloader:
            discriminator.zero_grad()
            generated = generator()
            pred_real = discriminator(minimizer)
            pred_gen = discriminator(generated)

            loss_D = loss_fn(pred_real, 1.0) + loss_fn(pred_gen, 0.0)
            loss_D.backward()
            losses_D.append(loss_D)
            opt_discrim.step()

            #TODO: multiple training
            generator.zero_grad()
            pred_gen = discriminator(generated)
            loss_G = loss_fn(pred_gen, 1.0)
            loss_G.backward()
            losses_G.append(loss_G)
            opt_gen.step()

        print("Epoch %d, Generator loss %f, Discriminator loss %f" % (epoch, np.mean(losses_D), np.mean(losses_G)))