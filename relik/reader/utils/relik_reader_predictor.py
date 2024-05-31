import logging
from typing import Iterable, Iterator, List, Optional

import hydra
import torch
from lightning.pytorch.utilities import move_data_to_device
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from relik.reader.data.patches import merge_patches_predictions
from relik.reader.data.relik_reader_sample import (
    RelikReaderSample,
    load_relik_reader_samples,
)
from relik.reader.pytorch_modules.base import RelikReaderBase
from relik.reader.utils.special_symbols import NME_SYMBOL

logger = logging.getLogger(__name__)


def convert_tokens_to_char_annotations(
    sample: RelikReaderSample, remove_nmes: bool = False
):
    char_annotations = set()

    for (
        predicted_entity,
        predicted_spans,
    ) in sample.predicted_window_labels.items():
        if predicted_entity == NME_SYMBOL and remove_nmes:
            continue

        for span_start, span_end in predicted_spans:
            span_start = sample.token2char_start[str(span_start)]
            span_end = sample.token2char_end[str(span_end)]

            char_annotations.add((span_start, span_end, predicted_entity))

    char_probs_annotations = dict()
    for (
        span_start,
        span_end,
    ), candidates_probs in sample.span_title_probabilities.items():
        span_start = sample.token2char_start[str(span_start)]
        span_end = sample.token2char_end[str(span_end)]
        char_probs_annotations[(span_start, span_end)] = {
            title for title, _ in candidates_probs
        }

    sample.predicted_window_labels_chars = char_annotations
    sample.probs_window_labels_chars = char_probs_annotations


class RelikReaderPredictor:
    def __init__(
        self,
        relik_reader_core: RelikReaderBase,
        dataset_conf: Optional[dict] = None,
        predict_nmes: bool = False,
        dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.relik_reader_core = relik_reader_core
        self.dataset_conf = dataset_conf
        self.predict_nmes = predict_nmes
        self.dataloader: DataLoader | None = dataloader

        if self.dataset_conf is not None and self.dataset is None:
            # instantiate dataset
            self.dataset = hydra.utils.instantiate(
                dataset_conf,
                dataset_path=None,
                samples=None,
            )

    def predict(
        self,
        path: Optional[str],
        samples: Optional[Iterable[RelikReaderSample]],
        dataset_conf: Optional[dict],
        token_batch_size: int = 1024,
        progress_bar: bool = False,
        **kwargs,
    ) -> List[RelikReaderSample]:
        annotated_samples = list(
            self._predict(path, samples, dataset_conf, token_batch_size, progress_bar)
        )
        for sample in annotated_samples:
            merge_patches_predictions(sample)
            convert_tokens_to_char_annotations(
                sample, remove_nmes=not self.predict_nmes
            )
        return annotated_samples

    def _predict(
        self,
        path: Optional[str],
        samples: Optional[Iterable[RelikReaderSample]],
        dataset_conf: dict,
        token_batch_size: int = 1024,
        progress_bar: bool = False,
        **kwargs,
    ) -> Iterator[RelikReaderSample]:
        assert (
            path is not None or samples is not None
        ), "Either predict on a path or on an iterable of samples"

        next_prediction_position = 0
        position2predicted_sample = {}

        if self.dataloader is not None:
            iterator = self.dataloader
            for i, sample in enumerate(self.dataloader.dataset.samples):
                sample._mixin_prediction_position = i
        else:
            samples = load_relik_reader_samples(path) if samples is None else samples

            # setup infrastructure to re-yield in order
            def samples_it():
                for i, sample in enumerate(samples):
                    assert sample._mixin_prediction_position is None
                    sample._mixin_prediction_position = i
                    yield sample

            # instantiate dataset
            if getattr(self, "dataset", None) is not None:
                dataset = self.dataset
                dataset.samples = samples_it()
                dataset.tokens_per_batch = token_batch_size
            else:
                dataset = hydra.utils.instantiate(
                    dataset_conf,
                    dataset_path=None,
                    samples=samples_it(),
                    tokens_per_batch=token_batch_size,
                )

            # instantiate dataloader
            iterator = DataLoader(
                dataset, batch_size=None, num_workers=0, shuffle=False
            )
        if progress_bar:
            iterator = tqdm(iterator, desc="Predicting")

        model_device = next(self.relik_reader_core.parameters()).device

        with torch.inference_mode():
            for batch in iterator:
                # do batch predict
                with torch.autocast(
                    "cpu" if model_device == torch.device("cpu") else "cuda"
                ):
                    batch = move_data_to_device(batch, model_device)
                    batch_out = self.relik_reader_core._batch_predict(**batch)
                # update prediction position position
                for sample in batch_out:
                    if sample._mixin_prediction_position >= next_prediction_position:
                        position2predicted_sample[
                            sample._mixin_prediction_position
                        ] = sample

                # yield
                while next_prediction_position in position2predicted_sample:
                    yield position2predicted_sample[next_prediction_position]
                    del position2predicted_sample[next_prediction_position]
                    next_prediction_position += 1

        if len(position2predicted_sample) > 0:
            logger.warning(
                "It seems samples have been discarded in your dataset. "
                "This means that you WON'T have a prediction for each input sample. "
                "Prediction order will also be partially disrupted"
            )
            for k, v in sorted(position2predicted_sample.items(), key=lambda x: x[0]):
                yield v

        if progress_bar:
            iterator.close()
