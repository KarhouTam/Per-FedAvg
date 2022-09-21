import rich
import torch
import utils
from copy import deepcopy
from typing import Tuple, Union
from collections import OrderedDict
from data.utils import get_dataloader
from fedlab.utils.serialization import SerializationTool


class PerFedAvgClient:
    def __init__(
        self,
        client_id: int,
        alpha: float,
        beta: float,
        global_model: torch.nn.Module,
        criterion: Union[torch.nn.CrossEntropyLoss, torch.nn.MSELoss],
        batch_size: int,
        dataset: str,
        local_epochs: int,
        valset_ratio: float,
        logger: rich.console.Console,
        gpu: int,
    ):
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

    def train(
        self,
        global_model: torch.nn.Module,
        hessian_free=False,
        eval_while_training=False,
    ):
        self.model.load_state_dict(global_model.state_dict())
        if eval_while_training:
            loss_before, acc_before = utils.eval(
                self.model, self.valloader, self.criterion, self.device
            )
        self._train(hessian_free)

        if eval_while_training:
            loss_after, acc_after = utils.eval(
                self.model, self.valloader, self.criterion, self.device
            )
            self.logger.log(
                "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%".format(
                    self.id,
                    loss_before,
                    loss_after,
                    acc_before * 100.0,
                    acc_after * 100.0,
                )
            )
        return SerializationTool.serialize_model(self.model)

    def _train(self, hessian_free=False):
        if hessian_free:  # Per-FedAvg(HF)
            for _ in range(self.local_epochs):
                temp_model = deepcopy(self.model)
                data_batch_1 = utils.get_data_batch(
                    self.trainloader, self.iter_trainloader, self.device
                )
                grads = self.compute_grad(temp_model, data_batch_1)
                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = utils.get_data_batch(
                    self.trainloader, self.iter_trainloader, self.device
                )
                grads_1st = self.compute_grad(temp_model, data_batch_2)

                data_batch_3 = utils.get_data_batch(
                    self.trainloader, self.iter_trainloader, self.device
                )

                grads_2nd = self.compute_grad(
                    self.model, data_batch_3, v=grads_1st, second_order_grads=True
                )
                # NOTE: Go check https://github.com/KarhouTam/Per-FedAvg/issues/2 if you confuse about the model update.
                for param, grad1, grad2 in zip(
                    self.model.parameters(), grads_1st, grads_2nd
                ):
                    param.data.sub_(self.beta * grad1 - self.beta * self.alpha * grad2)

        else:  # Per-FedAvg(FO)
            for _ in range(self.local_epochs):
                # ========================== FedAvg ==========================
                # NOTE: You can uncomment those codes for running FedAvg.
                #       When you're trying to run FedAvg, comment other codes in this branch.

                # data_batch = utils.get_data_batch(
                #     self.trainloader, self.iter_trainloader, self.device
                # )
                # grads = self.compute_grad(self.model, data_batch)
                # for param, grad in zip(self.model.parameters(), grads):
                #     param.data.sub_(self.beta * grad)

                # ============================================================

                temp_model = deepcopy(self.model)
                data_batch_1 = utils.get_data_batch(
                    self.trainloader, self.iter_trainloader, self.device
                )
                grads = self.compute_grad(temp_model, data_batch_1)

                for param, grad in zip(temp_model.parameters(), grads):
                    param.data.sub_(self.alpha * grad)

                data_batch_2 = utils.get_data_batch(
                    self.trainloader, self.iter_trainloader, self.device
                )
                grads = self.compute_grad(temp_model, data_batch_2)

                for param, grad in zip(self.model.parameters(), grads):
                    param.data.sub_(self.beta * grad)

    def compute_grad(
        self,
        model: torch.nn.Module,
        data_batch: Tuple[torch.Tensor, torch.Tensor],
        v: Union[Tuple[torch.Tensor, ...], None] = None,
        second_order_grads=False,
    ):
        x, y = data_batch
        if second_order_grads:
            frz_model_params = deepcopy(model.state_dict())
            delta = 1e-3
            dummy_model_params_1 = OrderedDict()
            dummy_model_params_2 = OrderedDict()
            with torch.no_grad():
                for (layer_name, param), grad in zip(model.named_parameters(), v):
                    dummy_model_params_1.update({layer_name: param + delta * grad})
                    dummy_model_params_2.update({layer_name: param - delta * grad})

            model.load_state_dict(dummy_model_params_1, strict=False)
            logit_1 = model(x)
            # loss_1 = self.criterion(logit_1, y) / y.size(-1)
            loss_1 = self.criterion(logit_1, y)
            grads_1 = torch.autograd.grad(loss_1, model.parameters())

            model.load_state_dict(dummy_model_params_2, strict=False)
            logit_2 = model(x)
            loss_2 = self.criterion(logit_2, y)
            # loss_2 = self.criterion(logit_2, y) / y.size(-1)
            grads_2 = torch.autograd.grad(loss_2, model.parameters())

            model.load_state_dict(frz_model_params)

            grads = []
            with torch.no_grad():
                for g1, g2 in zip(grads_1, grads_2):
                    grads.append((g1 - g2) / (2 * delta))
            return grads

        else:
            logit = model(x)
            # loss = self.criterion(logit, y) / y.size(-1)
            loss = self.criterion(logit, y)
            grads = torch.autograd.grad(loss, model.parameters())
            return grads

    def pers_N_eval(self, global_model: torch.nn.Module, pers_epochs: int):
        self.model.load_state_dict(global_model.state_dict())

        loss_before, acc_before = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha)
        for _ in range(pers_epochs):
            x, y = utils.get_data_batch(
                self.trainloader, self.iter_trainloader, self.device
            )
            logit = self.model(x)
            # loss = self.criterion(logit, y) / y.size(-1)
            loss = self.criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_after, acc_after = utils.eval(
            self.model, self.valloader, self.criterion, self.device
        )
        self.logger.log(
            "client [{}] [red]loss: {:.4f} -> {:.4f}   [blue]acc: {:.2f}% -> {:.2f}%".format(
                self.id, loss_before, loss_after, acc_before * 100.0, acc_after * 100.0,
            )
        )
        return {
            "loss_before": loss_before,
            "acc_before": acc_before,
            "loss_after": loss_after,
            "acc_after": acc_after,
        }

