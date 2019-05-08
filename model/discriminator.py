import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dim, args):
        if args.activation == "relu":
            self.activation = nn.ReLU()
        elif args.activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif args.activation == "tanh":
            self.activation = nn.Tanh()
        layers = []
        layers += nn.Linear(dim, dim * 2)
        layers += self.activation
        layers += nn.Linear(dim * 2, dim * 2)
        layers += self.activation
        layers += nn.Linear(dim * 2, dim * 4)
        layers += self.activation
        layers += nn.Linear(dim * 4, 1)
        layers += nn.Sigmoid()

        self.model = nn.Sequential(*layers)
        self.loss = nn.BCELoss()

    def forward(self, x, labels):
        logits = self.model(x)
        loss = self.loss(logits, labels)
        return logits, loss