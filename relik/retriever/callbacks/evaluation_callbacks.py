import logging
from typing import Dict, List, Optional

import lightning as pl
import torch
from lightning.pytorch.trainer.states import RunningStage
from sklearn.metrics import label_ranking_average_precision_score

from relik.common.log import get_logger
from relik.retriever.callbacks.base import DEFAULT_STAGES, NLPTemplateCallback

logger = get_logger(__name__, level=logging.INFO)


class RecallAtKEvaluationCallback(NLPTemplateCallback):
    """
    Computes the recall at k for the predictions. Recall at k is computed as the number of
    correct predictions in the top k predictions divided by the total number of correct
    predictions.

    Args:
        k (`int`):
            The number of predictions to consider.
        prefix (`str`, `optional`):
            The prefix to add to the metrics.
        verbose (`bool`, `optional`, defaults to `False`):
            Whether to log the metrics.
        prog_bar (`bool`, `optional`, defaults to `True`):
            Whether to log the metrics to the progress bar.
    """

    def __init__(
        self,
        k: int = 100,
        prefix: Optional[str] = None,
        verbose: bool = False,
        prog_bar: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.prefix = prefix
        self.verbose = verbose
        self.prog_bar = prog_bar

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        """
        Computes the recall at k for the predictions.

        Args:
            trainer (:obj:`lightning.trainer.trainer.Trainer`):
                The trainer object.
            pl_module (:obj:`lightning.core.lightning.LightningModule`):
                The lightning module.
            predictions (:obj:`Dict`):
                The predictions.

        Returns:
            :obj:`Dict`: The computed metrics.
        """
        if self.verbose:
            logger.info(f"Computing recall@{self.k}")

        # metrics to return
        metrics = {}

        stage = trainer.state.stage
        if stage not in DEFAULT_STAGES:
            raise ValueError(
                f"Stage {stage} not supported, only `validate` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            hits, total = 0, 0
            for sample in samples:
                # compute the recall at k
                # cut the predictions to the first k elements
                predictions = sample["predictions"][: self.k]
                hits += len(set(predictions) & set(sample["gold"]))
                total += len(set(sample["gold"]))

            # compute the mean recall at k
            recall_at_k = hits / total
            metrics[f"recall@{self.k}_{dataloader_idx}"] = recall_at_k
        metrics[f"recall@{self.k}"] = sum(metrics.values()) / len(metrics)

        if self.prefix is not None:
            metrics = {f"{self.prefix}_{k}": v for k, v in metrics.items()}
        else:
            metrics = {f"{stage.value}_{k}": v for k, v in metrics.items()}
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=self.prog_bar
        )

        if self.verbose:
            logger.info(
                f"Recall@{self.k} on {stage.value}: {metrics[f'{stage.value}_recall@{self.k}']}"
            )

        return metrics


class AvgRankingEvaluationCallback(NLPTemplateCallback):
    """
    Computes the average ranking of the gold label in the predictions. Average ranking is
    computed as the average of the rank of the gold label in the predictions.

    Args:
        k (`int`):
            The number of predictions to consider.
        prefix (`str`, `optional`):
            The prefix to add to the metrics.
        stages (`List[str]`, `optional`):
            The stages to compute the metrics on. Defaults to `["validate", "test"]`.
        verbose (`bool`, `optional`, defaults to `False`):
            Whether to log the metrics.
    """

    def __init__(
        self,
        k: int,
        prefix: Optional[str] = None,
        stages: Optional[List[str]] = None,
        verbose: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.prefix = prefix
        self.verbose = verbose
        self.stages = (
            [RunningStage(stage) for stage in stages] if stages else DEFAULT_STAGES
        )

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        """
        Computes the average ranking of the gold label in the predictions.

        Args:
            trainer (:obj:`lightning.trainer.trainer.Trainer`):
                The trainer object.
            pl_module (:obj:`lightning.core.lightning.LightningModule`):
                The lightning module.
            predictions (:obj:`Dict`):
                The predictions.

        Returns:
            :obj:`Dict`: The computed metrics.
        """
        if not predictions:
            logger.warning("No predictions to compute the AVG Ranking metrics.")
            return {}

        if self.verbose:
            logger.info(f"Computing AVG Ranking@{self.k}")

        # metrics to return
        metrics = {}

        stage = trainer.state.stage
        if stage not in self.stages:
            raise ValueError(
                f"Stage `{stage}` not supported, only `validate` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            rankings = []
            for sample in samples:
                window_candidates = sample["predictions"][: self.k]
                window_labels = sample["gold"]
                for wl in window_labels:
                    if wl in window_candidates:
                        rankings.append(window_candidates.index(wl) + 1)

            avg_ranking = sum(rankings) / len(rankings) if len(rankings) > 0 else 0
            metrics[f"avg_ranking@{self.k}_{dataloader_idx}"] = avg_ranking
        if len(metrics) == 0:
            metrics[f"avg_ranking@{self.k}"] = 0
        else:
            metrics[f"avg_ranking@{self.k}"] = sum(metrics.values()) / len(metrics)

        prefix = self.prefix or stage.value
        metrics = {
            f"{prefix}_{k}": torch.as_tensor(v, dtype=torch.float32)
            for k, v in metrics.items()
        }
        pl_module.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=False)

        if self.verbose:
            logger.info(
                f"AVG Ranking@{self.k} on {prefix}: {metrics[f'{prefix}_avg_ranking@{self.k}']}"
            )

        return metrics


class LRAPEvaluationCallback(NLPTemplateCallback):
    def __init__(
        self,
        k: int = 100,
        prefix: Optional[str] = None,
        verbose: bool = False,
        prog_bar: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.k = k
        self.prefix = prefix
        self.verbose = verbose
        self.prog_bar = prog_bar

    @torch.no_grad()
    def __call__(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        predictions: Dict,
        *args,
        **kwargs,
    ) -> dict:
        if self.verbose:
            logger.info(f"Computing recall@{self.k}")

        # metrics to return
        metrics = {}

        stage = trainer.state.stage
        if stage not in DEFAULT_STAGES:
            raise ValueError(
                f"Stage {stage} not supported, only `validate` and `test` are supported."
            )

        for dataloader_idx, samples in predictions.items():
            scores = [sample["scores"][: self.k] for sample in samples]
            golds = [sample["gold"] for sample in samples]

            # compute the mean recall at k
            lrap = label_ranking_average_precision_score(golds, scores)
            metrics[f"lrap@{self.k}_{dataloader_idx}"] = lrap
        metrics[f"lrap@{self.k}"] = sum(metrics.values()) / len(metrics)

        prefix = self.prefix or stage.value
        metrics = {
            f"{prefix}_{k}": torch.as_tensor(v, dtype=torch.float32)
            for k, v in metrics.items()
        }
        pl_module.log_dict(
            metrics, on_step=False, on_epoch=True, prog_bar=self.prog_bar
        )

        if self.verbose:
            logger.info(
                f"Recall@{self.k} on {stage.value}: {metrics[f'{stage.value}_recall@{self.k}']}"
            )

        return metrics
