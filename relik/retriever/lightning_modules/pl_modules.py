from typing import Any, Union

import hydra
import lightning as pl
import torch
from omegaconf import DictConfig

from relik.retriever.common.model_inputs import ModelInputs


class GoldenRetrieverPLModule(pl.LightningModule):
    def __init__(
        self,
        model: Union[torch.nn.Module, DictConfig],
        optimizer: Union[torch.optim.Optimizer, DictConfig],
        lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, DictConfig] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        if isinstance(model, DictConfig):
            self.model = hydra.utils.instantiate(model)
        else:
            self.model = model

        self.optimizer_config = optimizer
        self.lr_scheduler_config = lr_scheduler

    def forward(self, **kwargs) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        return self.model(**kwargs)

    def training_step(self, batch: ModelInputs, batch_idx: int) -> torch.Tensor:
        forward_output = self.forward(**batch, return_loss=True)
        self.log(
            "loss",
            forward_output["loss"],
            batch_size=batch["questions"]["input_ids"].size(0),
            prog_bar=True,
        )
        return forward_output["loss"]

    def validation_step(self, batch: ModelInputs, batch_idx: int) -> None:
        forward_output = self.forward(**batch, return_loss=True)
        self.log(
            "val_loss",
            forward_output["loss"],
            batch_size=batch["questions"]["input_ids"].size(0),
        )

    def test_step(self, batch: ModelInputs, batch_idx: int) -> Any:
        forward_output = self.forward(**batch, return_loss=True)
        self.log(
            "test_loss",
            forward_output["loss"],
            batch_size=batch["questions"]["input_ids"].size(0),
        )

    def configure_optimizers(self):
        if isinstance(self.optimizer_config, DictConfig):
            param_optimizer = list(self.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if "layer_norm_layer" in n
                    ],
                    "weight_decay": self.hparams.optimizer.weight_decay,
                    "lr": 1e-4,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if all(nd not in n for nd in no_decay)
                        and "layer_norm_layer" not in n
                    ],
                    "weight_decay": self.hparams.optimizer.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if "layer_norm_layer" not in n
                        and any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = hydra.utils.instantiate(
                self.optimizer_config,
                # params=self.parameters(),
                params=optimizer_grouped_parameters,
                _convert_="partial",
            )
        else:
            optimizer = self.optimizer_config

        if self.lr_scheduler_config is None:
            return optimizer

        if isinstance(self.lr_scheduler_config, DictConfig):
            lr_scheduler = hydra.utils.instantiate(
                self.lr_scheduler_config, optimizer=optimizer
            )
        else:
            lr_scheduler = self.lr_scheduler_config

        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [lr_scheduler_config]
