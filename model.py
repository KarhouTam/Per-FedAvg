import torch.nn as nn
import torch
import torch.nn.functional as F


class elu(nn.Module):
    def __init__(self) -> None:
        super(elu, self).__init__()

    def forward(self, x):
        return torch.where(x >= 0, x, 0.2 * (torch.exp(x) - 1))


class linear(nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super(linear, self).__init__()
        self.w = nn.Parameter(
            torch.randn(out_c, in_c) * torch.sqrt(torch.tensor(2 / in_c))
        )
        self.b = nn.Parameter(torch.randn(out_c))

    def forward(self, x):
        return F.linear(x, self.w, self.b)


class MLP_MNIST(nn.Module):
    def __init__(self) -> None:
        super(MLP_MNIST, self).__init__()
        self.fc1 = linear(28 * 28, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x


class MLP_CIFAR10(nn.Module):
    def __init__(self) -> None:
        super(MLP_CIFAR10, self).__init__()
        self.fc1 = linear(32 * 32 * 3, 80)
        self.fc2 = linear(80, 60)
        self.fc3 = linear(60, 10)
        self.flatten = nn.Flatten()
        self.activation = elu()

    def forward(self, x):
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.activation(x)

        x = self.fc2(x)
        x = self.activation(x)

        x = self.fc3(x)
        x = self.activation(x)

        return x


MODEL_DICT = {"mnist": MLP_MNIST, "cifar": MLP_CIFAR10}


def get_model(dataset, device):
    return MODEL_DICT[dataset]().to(device)

