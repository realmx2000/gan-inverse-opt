import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, dim, activation):
        super().__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky":
            self.activation == nn.LeakyReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        layers = []
        layers.append(nn.Linear(dim, dim * 2))
        layers.append(self.activation)
        layers.append(nn.Linear(dim * 2, dim * 2))
        layers.append(self.activation)
        layers.append(nn.Linear(dim * 2, dim * 4))
        layers.append(self.activation)
        layers.append(nn.Linear(dim * 4, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)
        self.loss = nn.BCELoss()

    def forward(self, x, labels):
        logits = self.model(x)
        logits = logits.squeeze(1)
        loss = self.loss(logits, labels)
        return logits, loss