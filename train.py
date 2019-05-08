from .models import *
from dataset import SyntheticDataset
from torch.utils.data import DataLoader
from torch.optim import SGD, Adam

def get_optimizer(model, opt, lr, momentum, reg):
    if opt == "nesterov":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg, nesterov=True)
    elif opt == "sgd":
        optimizer = SGD(model.parameters(), lr, momentum=momentum, weight_decay=reg)
    elif opt == "adam":
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=reg)

def train(args):
    generator = Generator(args.obj_type, args.constraint_type, args.num_constraints,
                          args.dim, args.solver, args.solve_schedule, args.lamb)
    discriminator = Discriminator(args.dim, args.activation)

    dataset = SyntheticDataset(args.obj_type, args.constraint_type, args.num_constraints, args.dim)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    opt_gen = get_optimizer(generator, args.optimizer, args.lr, args.momentum, args.reg)
    opt_discrim = get_optimizer(discriminator, args.optimizer, args.lr, args.momentum, args.reg)