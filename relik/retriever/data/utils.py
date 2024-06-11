import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Union

import numpy as np
import transformers as tr
from tqdm import tqdm


class HardNegativesManager:
    def __init__(
        self,
        tokenizer: tr.PreTrainedTokenizer,
        data: Union[List[Dict], os.PathLike, Dict[int, List]] = None,
        max_length: int = 64,
        batch_size: int = 1000,
        lazy: bool = False,
    ) -> None:
        self._db: dict = None
        self.tokenizer = tokenizer

        if data is None:
            self._db = {}
        else:
            if isinstance(data, Dict):
                self._db = data
            elif isinstance(data, os.PathLike):
                with open(data) as f:
                    self._db = json.load(f)
            else:
                raise ValueError(
                    f"Data type {type(data)} not supported, only Dict and os.PathLike are supported."
                )
        # add the tokenizer to the class for future use
        self.tokenizer = tokenizer

        # invert the db to have a passage -> sample_idx mapping
        self._passage_db = defaultdict(set)
        for sample_idx, passages in self._db.items():
            for passage in passages:
                self._passage_db[passage].add(sample_idx)

        self._passage_hard_negatives = {}
        if not lazy:
            # create a dictionary of passage -> hard_negative mapping
            batch_size = min(batch_size, len(self._passage_db))
            unique_passages = list(self._passage_db.keys())
            for i in tqdm(
                range(0, len(unique_passages), batch_size),
                desc="Tokenizing Hard Negatives",
            ):
                batch = unique_passages[i : i + batch_size]
                tokenized_passages = self.tokenizer(
                    batch,
                    max_length=max_length,
                    truncation=True,
                )
                for i, passage in enumerate(batch):
                    self._passage_hard_negatives[passage] = {
                        k: tokenized_passages[k][i] for k in tokenized_passages.keys()
                    }

    def __len__(self) -> int:
        return len(self._db)

    def __getitem__(self, idx: int) -> Dict:
        return self._db[idx]

    def __iter__(self):
        for sample in self._db:
            yield sample

    def __contains__(self, idx: int) -> bool:
        return idx in self._db

    def get(self, idx: int) -> List[str]:
        """Get the hard negatives for a given sample index."""
        if idx not in self._db:
            raise ValueError(f"Sample index {idx} not in the database.")

        passages = self._db[idx]

        output = []
        for passage in passages:
            if passage not in self._passage_hard_negatives:
                self._passage_hard_negatives[passage] = self._tokenize(passage)
            output.append(self._passage_hard_negatives[passage])

        return output

    def _tokenize(self, passage: str) -> Dict:
        return self.tokenizer(passage, max_length=self.max_length, truncation=True)


class NegativeSampler:
    def __init__(
        self, num_elements: int, probabilities: Optional[Union[List, np.ndarray]] = None
    ):
        if not isinstance(probabilities, np.ndarray):
            probabilities = np.array(probabilities)

        if probabilities is None:
            # probabilities should sum to 1
            probabilities = np.random.random(num_elements)
            probabilities /= np.sum(probabilities)
        self.probabilities = probabilities

    def __call__(
        self,
        sample_size: int,
        num_samples: int = 1,
        probabilities: np.array = None,
        exclude: List[int] = None,
    ) -> np.array:
        """
        Fast sampling of `sample_size` elements from `num_elements` elements.
        The sampling is done by randomly shifting the probabilities and then
        finding the smallest of the negative numbers. This is much faster than
        sampling from a multinomial distribution.

        Args:
            sample_size (`int`):
                number of elements to sample
            num_samples (`int`, optional):
                number of samples to draw. Defaults to 1.
            probabilities (`np.array`, optional):
                probabilities of each element. Defaults to None.
            exclude (`List[int]`, optional):
                indices of elements to exclude. Defaults to None.

        Returns:
            `np.array`: array of sampled indices
        """
        if probabilities is None:
            probabilities = self.probabilities

        if exclude is not None:
            probabilities[exclude] = 0
            # re-normalize?
            # probabilities /= np.sum(probabilities)

        # replicate probabilities as many times as `num_samples`
        replicated_probabilities = np.tile(probabilities, (num_samples, 1))
        # get random shifting numbers & scale them correctly
        random_shifts = np.random.random(replicated_probabilities.shape)
        random_shifts /= random_shifts.sum(axis=1)[:, np.newaxis]
        # shift by numbers & find largest (by finding the smallest of the negative)
        shifted_probabilities = random_shifts - replicated_probabilities
        sampled_indices = np.argpartition(shifted_probabilities, sample_size, axis=1)[
            :, :sample_size
        ]
        return sampled_indices
