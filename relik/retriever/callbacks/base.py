from functools import partial
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import hydra
import lightning as pl
import torch
from lightning.pytorch.trainer.states import RunningStage
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from relik.common.log import get_logger
from relik.retriever.data.base.datasets import BaseDataset

logger = get_logger(__name__)


STAGES_COMPATIBILITY_MAP = {
    "train": RunningStage.TRAINING,
    "val": RunningStage.VALIDATING,
    "test": RunningStage.TESTING,
}

DEFAULT_STAGES = {
    RunningStage.VALIDATING,
    RunningStage.TESTING,
    RunningStage.SANITY_CHECKING,
    RunningStage.PREDICTING,
}


class PredictionCallback(pl.Callback):
    def __init__(
        self,
        batch_size: int = 32,
        stages: Optional[Set[Union[str, RunningStage]]] = None,
        other_callbacks: Optional[
            Union[List[DictConfig], List["NLPTemplateCallback"]]
        ] = None,
        datasets: Optional[Union[DictConfig, BaseDataset]] = None,
        dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        # parameters
        self.batch_size = batch_size
        self.datasets = datasets
        self.dataloaders = dataloaders

        # callback initialization
        if stages is None:
            stages = DEFAULT_STAGES

        # compatibily stuff
        stages = {STAGES_COMPATIBILITY_MAP.get(stage, stage) for stage in stages}
        self.stages = [RunningStage(stage) for stage in stages]
        self.other_callbacks = other_callbacks or []
        for i, callback in enumerate(self.other_callbacks):
            if isinstance(callback, DictConfig):
                self.other_callbacks[i] = hydra.utils.instantiate(
                    callback, _recursive_=False
                )

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> Any:
        # it should return the predictions
        raise NotImplementedError

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ):
        predictions = self(trainer, pl_module)
        for callback in self.other_callbacks:
            callback(
                trainer=trainer,
                pl_module=pl_module,
                callback=self,
                predictions=predictions,
            )

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        predictions = self(trainer, pl_module)
        for callback in self.other_callbacks:
            callback(
                trainer=trainer,
                pl_module=pl_module,
                callback=self,
                predictions=predictions,
            )

    @staticmethod
    def _get_datasets_and_dataloaders(
        dataset: Optional[Union[Dataset, DictConfig]],
        dataloader: Optional[DataLoader],
        trainer: pl.Trainer,
        dataloader_kwargs: Optional[Dict[str, Any]] = None,
        collate_fn: Optional[Callable] = None,
        collate_fn_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dataset], List[DataLoader]]:
        """
        Get the datasets and dataloaders from the datamodule or from the dataset provided.

        Args:
            dataset (`Optional[Union[Dataset, DictConfig]]`):
                The dataset to use. If `None`, the datamodule is used.
            dataloader (`Optional[DataLoader]`):
                The dataloader to use. If `None`, the datamodule is used.
            trainer (`pl.Trainer`):
                The trainer that contains the datamodule.
            dataloader_kwargs (`Optional[Dict[str, Any]]`):
                The kwargs to pass to the dataloader.
            collate_fn (`Optional[Callable]`):
                The collate function to use.
            collate_fn_kwargs (`Optional[Dict[str, Any]]`):
                The kwargs to pass to the collate function.

        Returns:
            `Tuple[List[Dataset], List[DataLoader]]`: The datasets and dataloaders.
        """
        # if a dataset is provided, use it
        if dataset is not None:
            dataloader_kwargs = dataloader_kwargs or {}
            # get dataset
            if isinstance(dataset, DictConfig):
                dataset = hydra.utils.instantiate(dataset, _recursive_=False)
            datasets = [dataset] if not isinstance(dataset, list) else dataset
            if dataloader is not None:
                dataloaders = (
                    [dataloader] if isinstance(dataloader, DataLoader) else dataloader
                )
            else:
                collate_fn = collate_fn or partial(
                    datasets[0].collate_fn, **collate_fn_kwargs
                )
                dataloader_kwargs["collate_fn"] = collate_fn
                dataloaders = [DataLoader(datasets[0], **dataloader_kwargs)]
        else:
            # get the dataloaders and datasets from the datamodule
            datasets = (
                trainer.datamodule.test_datasets
                if trainer.state.stage == RunningStage.TESTING
                else trainer.datamodule.val_datasets
            )
            dataloaders = (
                trainer.test_dataloaders
                if trainer.state.stage == RunningStage.TESTING
                else trainer.val_dataloaders
            )
        return datasets, dataloaders


class NLPTemplateCallback:
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        callback: PredictionCallback,
        predictions: Dict[str, Any],
        *args,
        **kwargs,
    ) -> Any:
        raise NotImplementedError
