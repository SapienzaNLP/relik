import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning as pl
import torch
from lightning.pytorch.trainer.states import RunningStage

from relik.common.log import get_logger
from relik.retriever.callbacks.base import NLPTemplateCallback, PredictionCallback
from relik.retriever.pytorch_modules.hf import GoldenRetrieverModel

logger = get_logger(__name__, level=logging.INFO)


class SavePredictionsCallback(NLPTemplateCallback):
    def __init__(
        self,
        saving_dir: str | os.PathLike | None = None,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.saving_dir = saving_dir
        self.verbose = verbose

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Dict,
        callback: PredictionCallback,
        *args,
        **kwargs,
    ) -> dict:
        # write the predictions to a file inside the experiment folder
        if self.saving_dir is None and trainer.logger is None:
            logger.info(
                "You need to specify an output directory (`saving_dir`) or a logger to save the predictions.\n"
                "Skipping saving predictions."
            )
            return {}
        datasets = callback.datasets
        for dataloader_idx, dataloader_predictions in predictions.items():
            # save to file
            if self.saving_dir is not None:
                prediction_folder = Path(self.saving_dir)
            else:
                try:
                    prediction_folder = (
                        Path(trainer.logger.experiment.dir) / "predictions"
                    )
                except Exception:
                    logger.info(
                        "You need to specify an output directory (`saving_dir`) or a logger to save the predictions.\n"
                        "Skipping saving predictions."
                    )
                    return {}
                prediction_folder.mkdir(exist_ok=True)
            predictions_path = (
                prediction_folder
                / f"{datasets[dataloader_idx].name}_{dataloader_idx}.json"
            )
            if self.verbose:
                logger.info(f"Saving predictions to {predictions_path}")
            with open(predictions_path, "w") as f:
                for prediction in dataloader_predictions:
                    for k, v in prediction.items():
                        if isinstance(v, set):
                            prediction[k] = list(v)
                    f.write(json.dumps(prediction) + "\n")


class ResetModelCallback(pl.Callback):
    def __init__(
        self,
        question_encoder: str,
        passage_encoder: str | None = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.question_encoder = question_encoder
        self.passage_encoder = passage_encoder
        self.verbose = verbose

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        if trainer.current_epoch == 0:
            if self.verbose:
                logger.info("Current epoch is 0, skipping resetting model")
            return

        if self.verbose:
            logger.info("Resetting model, optimizer and lr scheduler")
        # reload model from scratch
        previous_device = pl_module.device
        trainer.model.model.question_encoder = GoldenRetrieverModel.from_pretrained(
            self.question_encoder
        )
        trainer.model.model.question_encoder.to(previous_device)
        if self.passage_encoder is not None:
            trainer.model.model.passage_encoder = GoldenRetrieverModel.from_pretrained(
                self.passage_encoder
            )
            trainer.model.model.passage_encoder.to(previous_device)

        trainer.strategy.setup_optimizers(trainer)


class FreeUpIndexerVRAMCallback(pl.Callback):
    def __call__(
        self,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ) -> Any:
        logger.info("Freeing up GPU memory")

        # remove the index from the GPU memory
        # remove the embeddings from the GPU memory first
        if pl_module.model.document_index is not None:
            if pl_module.model.document_index.embeddings is not None:
                try:
                    pl_module.model.document_index.embeddings.cpu()
                except Exception:
                    logger.warning(
                        "Could not move embeddings to CPU. Skipping freeing up VRAM."
                    )
                    pass
            pl_module.model.document_index.embeddings = None

        import gc

        gc.collect()
        torch.cuda.empty_cache()

    def on_train_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        return self(pl_module)

    def on_test_epoch_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, *args, **kwargs
    ) -> None:
        return self(pl_module)


class ShuffleTrainDatasetCallback(pl.Callback):
    def __init__(self, seed: int = 42, verbose: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.verbose = verbose
        self.previous_epoch = -1

    def on_validation_epoch_end(self, trainer: pl.Trainer, *args, **kwargs):
        if self.verbose:
            if trainer.current_epoch != self.previous_epoch:
                logger.info(f"Shuffling train dataset at epoch {trainer.current_epoch}")

            # logger.info(f"Shuffling train dataset at epoch {trainer.current_epoch}")
        if trainer.current_epoch != self.previous_epoch:
            trainer.datamodule.train_dataset.shuffle_data(
                seed=self.seed + trainer.current_epoch + 1
            )
            self.previous_epoch = trainer.current_epoch


class PrefetchTrainDatasetCallback(pl.Callback):
    def __init__(self, verbose: bool = True) -> None:
        super().__init__()
        self.verbose = verbose
        # self.previous_epoch = -1

    def on_validation_epoch_end(self, trainer: pl.Trainer, *args, **kwargs):
        if trainer.datamodule.train_dataset.prefetch_batches:
            if self.verbose:
                # if trainer.current_epoch != self.previous_epoch:
                logger.info(
                    f"Prefetching train dataset at epoch {trainer.current_epoch}"
                )
            # if trainer.current_epoch != self.previous_epoch:
            trainer.datamodule.train_dataset.prefetch()
            self.previous_epoch = trainer.current_epoch


class SubsampleTrainDatasetCallback(pl.Callback):
    def __init__(self, seed: int = 43, verbose: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.verbose = verbose

    def on_validation_epoch_end(self, trainer: pl.Trainer, *args, **kwargs):
        if self.verbose:
            logger.info(f"Subsampling train dataset at epoch {trainer.current_epoch}")
            trainer.datamodule.train_dataset.random_subsample(
                seed=self.seed + trainer.current_epoch + 1
            )


class SaveRetrieverCallback(pl.Callback):
    def __init__(
        self,
        saving_dir: str | os.PathLike | None = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.saving_dir = saving_dir
        self.verbose = verbose
        self.free_up_indexer_callback = FreeUpIndexerVRAMCallback()

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ):
        if self.saving_dir is None and trainer.logger is None:
            logger.info(
                "You need to specify an output directory (`saving_dir`) or a logger to save the retriever.\n"
                "Skipping saving retriever."
            )
            return
        if self.saving_dir is not None:
            retriever_folder = Path(self.saving_dir)
        else:
            try:
                retriever_folder = Path(trainer.logger.experiment.dir) / "retriever"
            except Exception:
                logger.info(
                    "You need to specify an output directory (`saving_dir`) or a logger to save the "
                    "retriever.\nSkipping saving retriever."
                )
                return
        retriever_folder.mkdir(exist_ok=True, parents=True)
        if self.verbose:
            logger.info(f"Saving retriever to {retriever_folder}")
        pl_module.model.save_pretrained(retriever_folder)

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: Dict[str, Any],
    ):
        self(trainer, pl_module)
        # self.free_up_indexer_callback(pl_module)


class SampleNegativesDatasetCallback(pl.Callback):
    def __init__(self, seed: int = 42, verbose: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.verbose = verbose

    def on_validation_epoch_end(self, trainer: pl.Trainer, *args, **kwargs):
        if self.verbose:
            f"Sampling negatives for train dataset at epoch {trainer.current_epoch}"
        trainer.datamodule.train_dataset.sample_dataset_negatives(
            seed=self.seed + trainer.current_epoch
        )


class SubsampleDataCallback(pl.Callback):
    def __init__(self, seed: int = 42, verbose: bool = True) -> None:
        super().__init__()
        self.seed = seed
        self.verbose = verbose

    def on_validation_epoch_start(self, trainer: pl.Trainer, *args, **kwargs):
        if self.verbose:
            f"Subsampling data for train dataset at epoch {trainer.current_epoch}"
        if trainer.state.stage == RunningStage.SANITY_CHECKING:
            return
        trainer.datamodule.train_dataset.subsample_data(
            seed=self.seed + trainer.current_epoch
        )

    def on_fit_start(self, trainer: pl.Trainer, *args, **kwargs):
        if self.verbose:
            f"Subsampling data for train dataset at epoch {trainer.current_epoch}"
        trainer.datamodule.train_dataset.subsample_data(
            seed=self.seed + trainer.current_epoch
        )
