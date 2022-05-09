import torch
import utils
from copy import deepcopy
from typing import Dict, Optional, OrderedDict, Tuple
from collections import OrderedDict
from data import get_dataloader
from fedlab.utils.serialization import SerializationTool


class PerFedAvgClient:
    def __init__(
        self,
        client_id,
        alpha,
        beta,
        global_model,
        criterion,
        batch_size,
        dataset,
        local_epochs,
        valset_ratio,
        logger,
        gpu,
    ) -> None:
        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.logger = logger

        self.local_epochs = local_epochs
        self.criterion = criterion
        self.id = client_id
        self.model = deepcopy(global_model)
        self.alpha = alpha
        self.beta = beta
        self.trainloader, self.valloader = get_dataloader(
            dataset, client_id, batch_size, valset_ratio
        )
        self.iter_trainloader = iter(self.trainloader)
        self.iter_valloader = iter(self.valloader)

    def train(
        self,
        global_model,
        epochs=None,
        hessian_free=False,
        eval_while_training=False,
        evaluation=False,
    ) -> Tuple[OrderedDict, Optional[Dict]]:
        self.model.load_state_dict(global_model.state_dict())
        if eval_while_training:
            loss_before, acc_before = utils.eval(
                self.model, self.valloader, self.criterion, self.device
            )
        _epochs = self.local_epochs if epochs is None else epochs
        self._train(_epochs, hessian_free, evaluation)

        if eval_while_training:
            loss_after, acc_after = utils.eval(
                self.model, self.valloader, self.criterion, self.device
            )
            self.logger.log(
                f"client [{self.id}] [red]loss: (before) {loss_before:.4f} (after) {loss_after:.4f}   [green]acc: (before) {(acc_before * 100.0):.2f}% (after) {(acc_after * 100.0):.2f}%"
            )
        if eval_while_training:
            return (
                SerializationTool.serialize_model(self.model),
                {
                    "loss_before": loss_before,
                    "acc_before": acc_before,
                    "loss_after": loss_after,
                    "acc_after": acc_after,
                },
            )
        else:
            return SerializationTool.serialize_model(self.model), None

    def _train(self, epochs, hessian_free=False, evaluation=False) -> None:
        if epochs <= 0:
            return

        if evaluation:
            dataloader = self.valloader
            iterator = self.iter_valloader
        else:
            dataloader = self.trainloader
            iterator = self.iter_trainloader

        if hessian_free:  # Per-FedAvg(HF)
            for _ in range(epochs):
                frz_model_params = deepcopy(self.model.state_dict())

                data_batch_1 = utils.get_data_batch(dataloader, iterator, self.device)
                grads = self.compute_grad(self.model, data_batch_1, False)
                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = utils.get_data_batch(dataloader, iterator, self.device)
                grads_1st = self.compute_grad(self.model, data_batch_2, False)

                data_batch_3 = utils.get_data_batch(dataloader, iterator, self.device)

                self.model.load_state_dict(frz_model_params)

                grads_2nd = self.compute_grad(self.model, data_batch_3, True)
                for g1, g2, param in zip(grads_1st, grads_2nd, self.model.parameters()):
                    final_grad = self.beta * (1.0 - g2) * g1
                    param.data.sub_(final_grad)
        else:  # Per-FedAvg(FO)
            for _ in range(epochs):
                frz_model_params = deepcopy(self.model.state_dict())

                data_batch_1 = utils.get_data_batch(dataloader, iterator, self.device)
                grads = self.compute_grad(self.model, data_batch_1, False)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = utils.get_data_batch(dataloader, iterator, self.device)
                grads = self.compute_grad(self.model, data_batch_2, False)

                self.model.load_state_dict(frz_model_params)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.beta * grad)

    def eval(self, global_model, pers_epochs, hessian_free=False) -> Dict:
        _, stats = self.train(
            global_model,
            epochs=pers_epochs,
            hessian_free=hessian_free,
            eval_while_training=True,
            evaluation=True,
        )
        return stats

    def compute_grad(
        self, model, data_batch, second_order_grads=False
    ) -> Tuple[torch.Tensor]:
        x, y = data_batch
        if second_order_grads:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-3
            logit = model(x)
            loss = self.criterion(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), grads):
                    dummy_model_params_1.update({layer_name: param + delta * 1})
                    dummy_model_params_2.update({layer_name: param - delta * 1})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            loss = self.criterion(logit_1, y)
            grads_1 = torch.autograd.grad(loss, model.parameters())

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_2 = model(x)
            loss = self.criterion(logit_2, y)
            grads_2 = torch.autograd.grad(loss, model.parameters())

            model.load_state_dict(frz_model_params)

            grads = []
            for u, v in zip(grads_1, grads_2):
                grads.append((u - v) / (2 * delta))
            return grads

        else:
            logit = model(x)
            loss = self.criterion(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads

    def get_data_batch(self, evaluation=False) -> Tuple[torch.Tensor, torch.Tensor]:
        if evaluation:
            try:
                x, y = next(self.iter_valloader)
            except StopIteration:
                self.iter_valloader = iter(self.valloader)
                x, y = next(self.iter_valloader)
        else:
            try:
                x, y = next(self.iter_trainloader)
            except StopIteration:
                self.iter_trainloader = iter(self.trainloader)
                x, y = next(self.iter_trainloader)

        return x.to(self.device), y.to(self.device)
