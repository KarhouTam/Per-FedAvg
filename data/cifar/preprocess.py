import os
import pickle
import numpy as np
import random
from argparse import ArgumentParser
from fedlab.utils.dataset.slicing import noniid_slicing
from torchvision.datasets import CIFAR10
from torchvision import transforms
from path import Path
from cifar import CIFARDataset

current_dir = Path(__file__).parent.abspath()


def preprocess(args):
    pickles_dir = current_dir / "pickles"
    raw_data_dir = current_dir / "raw_data"
    np.random.seed(args.seed)
    if os.path.isdir(pickles_dir):
        os.system(f"rm -rf {pickles_dir}")
    os.mkdir(f"{pickles_dir}")

    num_train_clients = int(args.client_num_in_total * args.fraction)
    num_test_clients = args.client_num_in_total - num_train_clients

    cifar10_train = CIFAR10(
        raw_data_dir, train=True, transform=transforms.ToTensor(), download=True,
    )
    cifar10_test = CIFAR10(raw_data_dir, train=False, transform=transforms.ToTensor(),)
    train_idxs = noniid_slicing(
        cifar10_train, num_train_clients, args.classes * num_train_clients,
    )

    test_idxs = noniid_slicing(
        cifar10_test, num_test_clients, args.classes * num_test_clients,
    )

    all_trainsets = []
    all_testsets = []

    for train_indices in train_idxs.values():
        all_trainsets.append(CIFARDataset([cifar10_train[i] for i in train_indices]))
    for test_indices in test_idxs.values():
        all_testsets.append(CIFARDataset([cifar10_test[i] for i in test_indices]))
    all_datasets = all_trainsets + all_testsets

    for client_id, dataset in enumerate(all_datasets):
        with open(pickles_dir / str(client_id) + ".pkl", "wb") as f:
            pickle.dump(dataset, f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--client_num_in_total", type=int, default=100)
    parser.add_argument(
        "--fraction", type=float, default=0.8, help="Propotion of train clients"
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
