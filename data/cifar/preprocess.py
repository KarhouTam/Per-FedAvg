import os
import pickle
import numpy as np
import random
from argparse import ArgumentParser
from fedlab.utils.dataset.slicing import noniid_slicing
from torchvision.datasets import CIFAR10
from torchvision import transforms
from path import Path

# from cifar import CIFARDataset

current_dir = Path(__file__).parent.abspath()

import torch
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    def __init__(self, subset) -> None:
        self.data = torch.stack(list(map(lambda tup: tup[0], subset)))
        self.targets = torch.stack(
            list(map(lambda tup: torch.tensor(tup[1], dtype=torch.long), subset))
        )
        self.transform = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.targets[index]

    def __len__(self):
        return len(self.targets)


def preprocess(args):
    pickles_dir = current_dir / "pickles"
    np.random.seed(args.seed)
    if os.path.isdir(pickles_dir):
        os.system(f"rm -rf {pickles_dir}")

    os.mkdir(pickles_dir)

    num_train_clients = int(args.client_num_in_total * args.fraction)
    num_test_clients = args.client_num_in_total - num_train_clients

    cifar_train = CIFAR10(current_dir, train=True, download=True,)
    cifar_test = CIFAR10(current_dir, train=False)
    train_idxs = noniid_slicing(
        cifar_train, num_train_clients, args.classes * num_train_clients,
    )

    test_idxs = noniid_slicing(
        cifar_test, num_test_clients, args.classes * num_test_clients,
    )

    all_trainsets = []
    all_testsets = []

    for train_indices in train_idxs.values():
        all_trainsets.append(CIFARDataset([cifar_train[i] for i in train_indices]))
    for test_indices in test_idxs.values():
        all_testsets.append(CIFARDataset([cifar_test[i] for i in test_indices]))
    all_datasets = all_trainsets + all_testsets

    for client_id, dataset in enumerate(all_datasets):
        with open(pickles_dir / str(client_id) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)
    with open(pickles_dir / "seperation.pkl", "wb") as f:
        pickle.dump(
            {
                "train": [i for i in range(num_train_clients)],
                "test": [i for i in range(num_train_clients, args.client_num_in_total)],
                "total": args.client_num_in_total,
            },
            f,
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument(
        "--fraction", type=float, default=0.9, help="Propotion of train clients"
    )
    parser.add_argument(
        "--classes",
        type=int,
        choices=[i for i in range(1, 11)],
        default=2,
        help="Num of classes of data that one client could have.",
    )
    parser.add_argument("--seed", type=int, default=random.seed())
    args = parser.parse_args()
    preprocess(args)
