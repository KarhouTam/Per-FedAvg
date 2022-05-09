import os
import pickle
from path import Path
from torch.utils.data import DataLoader
from .mnist import MNISTDataset
from torch.utils.data import random_split

current_dir = Path(__file__).parent.abspath()


def get_mnist_dataloader(client_id, batch_size=10, valset_ratio=0.1):
    pickles_dir = current_dir / "pickles"

    if os.path.isdir(pickles_dir) is False:
        raise RuntimeError("Please preprocess and create pickles first.")

    with open(pickles_dir / str(client_id) + ".pkl", "rb") as f:
        dataset: MNISTDataset = pickle.load(f)

    val_num_samples = int(valset_ratio * len(dataset))
    train_num_samples = len(dataset) - val_num_samples

    trainset, valset = random_split(dataset, [train_num_samples, val_num_samples])
    trainloader = DataLoader(trainset, batch_size, drop_last=True)
    valloader = DataLoader(valset, batch_size)

    return trainloader, valloader
