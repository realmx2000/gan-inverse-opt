import argparse

class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--prob_type", type=str, choices=["LP", "QP", "QCQP", "SOCP"], default="LP")
        self.parser.add_argument("--num_constraints", type=int, default=11)
        self.parser.add_argument("--constraint_dims", type=int, nargs="*",
                                 help="Dimension of each SOCP constraint. (Required for SOCP problems)")
        self.parser.add_argument("--dim", type=int, default=10, help="Dimensionality of solution space.")
        self.parser.add_argument("--lambda", type=float, default=1000.0, dest="lamb", help="Scale factor for log barrier.")
        self.parser.add_argument("--activation", type=str, choices=["relu", "leaky", "sigmoid", "tanh"], default="relu")
        self.parser.add_argument("--solver", type=str, choices=["subgradient", "newton"], default="newton",
                                 help="Solver to use in generator.")
        self.parser.add_argument("--solve_schedule", default="inv sq root", help="Step schedule for solver.")
        self.parser.add_argument("--optimizer", type=str, choices=["sgd", "nesterov", "adam"], default="adam")
        self.parser.add_argument("--gen_scheduler", type=str, choices=["none", "step", "cos", "plateau"], default="none")
        self.parser.add_argument("--dis_scheduler", type=str, choices=["none", "step", "cos", "plateau"], default="none")
        self.parser.add_argument("--momentum", type=float, default=0.1)
        self.parser.add_argument("--reg", type=float, default=1e-4)
        self.parser.add_argument("--lr_g", type=float, default=1e-2)
        self.parser.add_argument("--lr_d", type=float, default=1e-2)
        self.parser.add_argument("--gen_decay_step", type=float, default=10)
        self.parser.add_argument("--dis_decay_step", type=float, default=10)
        self.parser.add_argument("--gen_decay_rate", type=float, default=0.1)
        self.parser.add_argument("--dis_decay_rate", type=float, default=0.1)
        self.parser.add_argument("--gen_restart", type=float, default=20)
        self.parser.add_argument("--dis_restart", type=float, default=20)
        self.parser.add_argument("--gen_patience", type=float, default=10)
        self.parser.add_argument("--dis_patience", type=float, default=10)
        self.parser.add_argument("--epochs", type=int, default=1000)
        self.parser.add_argument("--batch_size", type=int, default=1)
        self.parser.add_argument("--train_ratio", type=int, default=4,
                                 help="Number of discriminator training steps per generator training step.")

    def parse_args(self):
        args = self.parser.parse_args()

        if args.solve_schedule.isdigit():
            args.solve_schedule = int(args.solve_schedule)

        return args

