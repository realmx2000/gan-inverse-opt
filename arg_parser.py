import argparse

class ArgParser(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--obj_type", type=str, choices=["linear"], default="linear")
        self.parser.add_argument("--constraint_type", type=str, choices=["linear"], default="linear")
        self.parser.add_argument("--num_constraints", type=int, default=11)
        self.parser.add_argument("--dim", type=int, default=10, help="Dimensionality of solution space.")
        self.parser.add_argument("--lambda", type=float, default=1.0, help="Scale factor for log barrier.")
        self.parser.add_argument("--activation", type=str, choices=["relu", "leaky", "sigmoid", "tanh"], default="relu")
        self.parser.add_argument("--solver", type=str, choices=["subgradident"], default="subgradient",
                                 help="Solver to use in generator.")
        self.parser.add_argument("--solve_schedule", default="inv", help="Step schedule for solver.")
        self.parser.add_argument("--optimizer", type=str, choices=["sgd", "nesterov", "adam"], default="adam")
        self.parser.add_argument("--momentum", type=float, default=0.1)
        self.parser.add_argument("--reg", type=float, default=1e-4)
        self.parser.add_argument("--lr", type=float, default=1e-3)
        self.parser.add_argument("--epochs", type=int, default=100)
        self.parser.add_argument("--batch_size", type=int, default=5)
        self.parser.add_argument("--train_ratio", type=int, default=4,
                                 help="Number of discriminator training steps per generator training step.")

    def parse_args(self):
        args = self.parser.parse_args()

        if args.solve_schedule.isdigit():
            args.solve_schedule = int(args.solve_schedule)

        return args

