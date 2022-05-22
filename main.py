import torch
import random
import os
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.aggregator import Aggregators
from rich.console import Console
from utils import get_args, init_random_seed
from model import get_model
from perfedavg import PerFedAvgClient
from data import get_client_id_indices

# ================== Can not remove these modules ===================
# these modules are imported for pickles.load deserializing properly.
from data.cifar import CIFARDataset
from data.mnist import MNISTDataset
# ===================================================================

if __name__ == "__main__":
    args = get_args()
    init_random_seed(args.seed)
    if os.path.isdir("./log") == False:
        os.mkdir("./log")
    if args.gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    global_model = get_model(args.dataset, device)
    logger = Console(record=True)
    logger.log(log_locals=True)
    clients_4_training, clients_4_eval, client_num_in_total = get_client_id_indices(
        args.dataset
    )

    # init clients
    clients = [
        PerFedAvgClient(
            client_id=client_id,
            alpha=args.alpha,
            beta=args.beta,
            global_model=global_model,
            criterion=torch.nn.CrossEntropyLoss(),
            batch_size=args.batch_size,
            dataset=args.dataset,
            local_epochs=args.local_epochs,
            valset_ratio=args.valset_ratio,
            logger=logger,
            gpu=args.gpu,
        )
        for client_id in range(client_num_in_total)
    ]
    # training
    logger.log("=" * 20, "TRAINING", "=" * 20, style="bold red")
    for _ in range(args.global_epochs):
        # select clients
        selected_clients = random.sample(clients_4_training, args.client_num_per_round)

        model_params_cache = []
        # client local training
        for client_id in selected_clients:
            serialized_model_params, _ = clients[client_id].train(
                global_model=global_model,
                hessian_free=args.hf,
                eval_while_training=args.eval_while_training,
            )
            model_params_cache.append(serialized_model_params)

        # aggregate model parameters
        aggregated_model_params = Aggregators.fedavg_aggregate(model_params_cache)
        SerializationTool.deserialize_model(global_model, aggregated_model_params)

    # eval
    logger.log("=" * 20, "EVALUATION", "=" * 20, style="bold blue")
    loss_before = []
    loss_after = []
    acc_before = []
    acc_after = []
    for client_id in clients_4_eval:
        stats = clients[client_id].eval(
            global_model=global_model,
            pers_epochs=args.pers_epochs,
            hessian_free=args.hf,
        )
        loss_before.append(stats["loss_before"])
        loss_after.append(stats["loss_after"])
        acc_before.append(stats["acc_before"])
        acc_after.append(stats["acc_after"])

    logger.log("=" * 20, "RESULTS", "=" * 20, style="bold green")
    logger.log(f"loss_before_pers: {(sum(loss_before) / len(loss_before)):.4f}")
    logger.log(f"acc_before_pers: {(sum(acc_before) * 100.0 / len(acc_before)):.2f}%")
    logger.log(f"loss_after_pers: {(sum(loss_after) / len(loss_after)):.4f}")
    logger.log(f"acc_after_pers: {(sum(acc_after) * 100.0 / len(acc_after)):.2f}%")

    algo = "HF" if args.hf else "FO"
    logger.save_html(
        f"./log/{args.dataset}_{args.client_num_per_round}_{args.global_epochs}_{args.pers_epochs}_{algo}.html"
    )

