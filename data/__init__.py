from .mnist import get_mnist_dataloader
from .cifar import get_cifar_dataloader
from path import Path
import os

FUNC_DICT = {
    "mnist": get_mnist_dataloader,
    "cifar": get_cifar_dataloader,
}

CURRENT_PATH = Path(__file__).parent.abspath()


def get_dataloader(dataset, client_id, batch_size=10, valset_ratio=0.1):
    if valset_ratio > 1 or valset_ratio < 0:
        raise ValueError("Wrong value of valset ratio.")
    return FUNC_DICT[dataset](client_id, batch_size, valset_ratio)


def get_client_id_indices(dataset, fraction):
    dataset_pickles_path = CURRENT_PATH / dataset / "pickles"
    client_num_in_total = len(os.listdir(dataset_pickles_path))
    clients_num_4_training = int(client_num_in_total * fraction)
    return (
        [i for i in range(clients_num_4_training)],
        [i for i in range(clients_num_4_training, client_num_in_total)],
        client_num_in_total,
    )

