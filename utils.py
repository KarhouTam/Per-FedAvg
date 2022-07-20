import torch
import random
import numpy as np
from typing import Iterator, Tuple, Union
from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--global_epochs", type=int, default=200)
    parser.add_argument("--local_epochs", type=int, default=4)
    parser.add_argument(
        "--pers_epochs",
        type=int,
        default=1,
        help="Indicate how many data batches would be used for personalization. Negatives means that equal to train phase.",
    )
    parser.add_argument(
        "--hf",
        type=int,
        default=1,
        help="0 for performing Per-FedAvg(FO), others for Per-FedAvg(HF)",
    )
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument(
        "--valset_ratio",
        type=float,
        default=0.1,
        help="Proportion of val set in the entire client local dataset",
    )
    parser.add_argument(
        "--dataset", type=str, choices=["mnist", "cifar"], default="mnist"
    )
    parser.add_argument("--client_num_per_round", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--gpu",
        type=int,
        default=1,
        help="Non-zero value for using gpu, 0 for using cpu",
    )
    parser.add_argument(
        "--eval_while_training",
        type=int,
        default=1,
        help="Non-zero value for performing local evaluation before and after local training",
    )
    parser.add_argument("--log", type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def eval(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: Union[torch.nn.MSELoss, torch.nn.CrossEntropyLoss],
    device=torch.device("cpu"),
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0
    num_samples = 0
    acc = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        logit = model(x)
        # total_loss += criterion(logit, y) / y.size(-1)
        total_loss += criterion(logit, y)
        pred = torch.softmax(logit, -1).argmax(-1)
        acc += torch.eq(pred, y).int().sum()
        num_samples += y.size(-1)
    model.train()
    return total_loss, acc / num_samples


def get_data_batch(
    dataloader: torch.utils.data.DataLoader,
    iterator: Iterator,
    device=torch.device("cpu"),
):
    try:
        x, y = next(iterator)
    except StopIteration:
        iterator = iter(dataloader)
        x, y = next(iterator)

    return x.to(device), y.to(device)


def fix_random_seed(seed: int):
    torch.cuda.empty_cache()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
