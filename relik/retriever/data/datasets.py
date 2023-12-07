import os
from copy import deepcopy
from enum import Enum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import datasets
import psutil
import torch
import transformers as tr
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm

from relik.common.log import get_logger
from relik.retriever.common.model_inputs import ModelInputs
from relik.retriever.data.base.datasets import BaseDataset, IterableBaseDataset
from relik.retriever.data.utils import HardNegativesManager

logger = get_logger(__name__)


class SubsampleStrategyEnum(Enum):
    NONE = "none"
    RANDOM = "random"
    IN_ORDER = "in_order"


class GoldenRetrieverDataset:
    def __init__(
        self,
        name: str,
        path: Union[str, os.PathLike, List[str], List[os.PathLike]] = None,
        data: Any = None,
        tokenizer: Optional[Union[str, tr.PreTrainedTokenizer]] = None,
        # passages: Union[str, os.PathLike, List[str]] = None,
        passage_batch_size: int = 32,
        question_batch_size: int = 32,
        max_positives: int = -1,
        max_negatives: int = 0,
        max_hard_negatives: int = 0,
        max_question_length: int = 256,
        max_passage_length: int = 64,
        shuffle: bool = False,
        subsample_strategy: Optional[str] = SubsampleStrategyEnum.NONE,
        subsample_portion: float = 0.1,
        num_proc: Optional[int] = None,
        load_from_cache_file: bool = True,
        keep_in_memory: bool = False,
        prefetch: bool = True,
        load_fn_kwargs: Optional[Dict[str, Any]] = None,
        batch_fn_kwargs: Optional[Dict[str, Any]] = None,
        collate_fn_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if path is None and data is None:
            raise ValueError("Either `path` or `data` must be provided")

        if tokenizer is None:
            raise ValueError("A tokenizer must be provided")

        # dataset parameters
        self.name = name
        self.path = Path(path) or path
        if path is not None and not isinstance(self.path, Sequence):
            self.path = [self.path]
        # self.project_folder = Path(__file__).parent.parent.parent
        self.data = data

        # hyper-parameters
        self.passage_batch_size = passage_batch_size
        self.question_batch_size = question_batch_size
        self.max_positives = max_positives
        self.max_negatives = max_negatives
        self.max_hard_negatives = max_hard_negatives
        self.max_question_length = max_question_length
        self.max_passage_length = max_passage_length
        self.shuffle = shuffle
        self.num_proc = num_proc
        self.load_from_cache_file = load_from_cache_file
        self.keep_in_memory = keep_in_memory
        self.prefetch = prefetch

        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = tr.AutoTokenizer.from_pretrained(self.tokenizer)

        self.padding_ops = {
            "input_ids": partial(
                self.pad_sequence,
                value=self.tokenizer.pad_token_id,
            ),
            "attention_mask": partial(self.pad_sequence, value=0),
            "token_type_ids": partial(
                self.pad_sequence,
                value=self.tokenizer.pad_token_type_id,
            ),
        }

        # check if subsample strategy is valid
        if subsample_strategy is not None:
            # subsample_strategy can be a string or a SubsampleStrategy
            if isinstance(subsample_strategy, str):
                try:
                    subsample_strategy = SubsampleStrategyEnum(subsample_strategy)
                except ValueError:
                    raise ValueError(
                        f"Subsample strategy `{subsample_strategy}` is not valid. "
                        f"Valid strategies are: {SubsampleStrategyEnum.__members__}"
                    )
            if not isinstance(subsample_strategy, SubsampleStrategyEnum):
                raise ValueError(
                    f"Subsample strategy `{subsample_strategy}` is not valid. "
                    f"Valid strategies are: {SubsampleStrategyEnum.__members__}"
                )
        self.subsample_strategy = subsample_strategy
        self.subsample_portion = subsample_portion

        # load the dataset
        if data is None:
            self.data: Dataset = self.load(
                self.path,
                tokenizer=self.tokenizer,
                load_from_cache_file=load_from_cache_file,
                load_fn_kwargs=load_fn_kwargs,
                num_proc=num_proc,
                shuffle=shuffle,
                keep_in_memory=keep_in_memory,
                max_positives=max_positives,
                max_negatives=max_negatives,
                max_hard_negatives=max_hard_negatives,
                max_question_length=max_question_length,
                max_passage_length=max_passage_length,
            )
        else:
            self.data: Dataset = data

        self.hn_manager: Optional[HardNegativesManager] = None

        # keep track of how many times the dataset has been iterated over
        self.number_of_complete_iterations = 0

    def __repr__(self) -> str:
        return f"GoldenRetrieverDataset({self.name=}, {self.path=})"

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError

    def to_torch_dataset(self, *args, **kwargs) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def load(
        self,
        paths: Union[str, os.PathLike, List[str], List[os.PathLike]],
        tokenizer: tr.PreTrainedTokenizer = None,
        load_fn_kwargs: Dict = None,
        load_from_cache_file: bool = True,
        num_proc: Optional[int] = None,
        shuffle: bool = False,
        keep_in_memory: bool = True,
        max_positives: int = -1,
        max_negatives: int = -1,
        max_hard_negatives: int = -1,
        max_passages: int = -1,
        max_question_length: int = 256,
        max_passage_length: int = 64,
        *args,
        **kwargs,
    ) -> Any:
        # if isinstance(paths, Sequence):
        #     paths = [self.project_folder / path for path in paths]
        # else:
        #     paths = [self.project_folder / paths]

        # read the data and put it in a placeholder list
        for path in paths:
            if not path.exists():
                raise ValueError(f"{path} does not exist")

        fn_kwargs = dict(
            tokenizer=tokenizer,
            max_positives=max_positives,
            max_negatives=max_negatives,
            max_hard_negatives=max_hard_negatives,
            max_passages=max_passages,
            max_question_length=max_question_length,
            max_passage_length=max_passage_length,
        )
        if load_fn_kwargs is not None:
            fn_kwargs.update(load_fn_kwargs)

        if num_proc is None:
            num_proc = psutil.cpu_count(logical=False)

        # The data is a list of dictionaries, each dictionary is a sample
        # Each sample has the following keys:
        #   - "question": the question
        #   - "answers": a list of answers
        #   - "positive_ctxs": a list of positive passages
        #   - "negative_ctxs": a list of negative passages
        #   - "hard_negative_ctxs": a list of hard negative passages
        # use the huggingface dataset library to load the data, by default it will load the
        # data in a dict with the key being "train".
        logger.info(f"Loading data for dataset {self.name}")
        data = load_dataset(
            "json",
            data_files=[str(p) for p in paths],  # datasets needs str paths and not Path
            split="train",
            streaming=False,  # TODO maybe we can make streaming work
            keep_in_memory=keep_in_memory,
        )
        # add id if not present
        if isinstance(data, datasets.Dataset):
            data = data.add_column("sample_idx", range(len(data)))
        else:
            data = data.map(
                lambda x, idx: x.update({"sample_idx": idx}), with_indices=True
            )

        map_kwargs = dict(
            function=self.load_fn,
            fn_kwargs=fn_kwargs,
        )
        if isinstance(data, datasets.Dataset):
            map_kwargs.update(
                dict(
                    load_from_cache_file=load_from_cache_file,
                    keep_in_memory=keep_in_memory,
                    num_proc=num_proc,
                    desc="Loading data",
                )
            )
        # preprocess the data
        data = data.map(**map_kwargs)

        # shuffle the data
        if shuffle:
            data.shuffle(seed=42)

        return data

    @staticmethod
    def create_batches(
        data: Dataset,
        batch_fn: Callable,
        batch_fn_kwargs: Optional[Dict[str, Any]] = None,
        prefetch: bool = True,
        *args,
        **kwargs,
    ) -> Union[Iterable, List]:
        if not prefetch:
            # if we are streaming, we don't need to create batches right now
            # we will create them on the fly when we need them
            batched_data = (
                batch
                for batch in batch_fn(
                    data, **(batch_fn_kwargs if batch_fn_kwargs is not None else {})
                )
            )
        else:
            batched_data = [
                batch
                for batch in tqdm(
                    batch_fn(
                        data, **(batch_fn_kwargs if batch_fn_kwargs is not None else {})
                    ),
                    desc="Creating batches",
                )
            ]
        return batched_data

    @staticmethod
    def collate_batches(
        batched_data: Union[Iterable, List],
        collate_fn: Callable,
        collate_fn_kwargs: Optional[Dict[str, Any]] = None,
        prefetch: bool = True,
        *args,
        **kwargs,
    ) -> Union[Iterable, List]:
        if not prefetch:
            collated_data = (
                collate_fn(batch, **(collate_fn_kwargs if collate_fn_kwargs else {}))
                for batch in batched_data
            )
        else:
            collated_data = [
                collate_fn(batch, **(collate_fn_kwargs if collate_fn_kwargs else {}))
                for batch in tqdm(batched_data, desc="Collating batches")
            ]
        return collated_data

    @staticmethod
    def load_fn(sample: Dict, *args, **kwargs) -> Dict:
        raise NotImplementedError

    @staticmethod
    def batch_fn(data: Dataset, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def collate_fn(batch: Any, *args, **kwargs) -> Any:
        raise NotImplementedError

    @staticmethod
    def pad_sequence(
        sequence: Union[List, torch.Tensor],
        length: int,
        value: Any = None,
        pad_to_left: bool = False,
    ) -> Union[List, torch.Tensor]:
        """
        Pad the input to the specified length with the given value.

        Args:
            sequence (:obj:`List`, :obj:`torch.Tensor`):
                Element to pad, it can be either a :obj:`List` or a :obj:`torch.Tensor`.
            length (:obj:`int`, :obj:`str`, optional, defaults to :obj:`subtoken`):
                Length after pad.
            value (:obj:`Any`, optional):
                Value to use as padding.
            pad_to_left (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True`, pads to the left, right otherwise.

        Returns:
            :obj:`List`, :obj:`torch.Tensor`: The padded sequence.

        """
        padding = [value] * abs(length - len(sequence))
        if isinstance(sequence, torch.Tensor):
            if len(sequence.shape) > 1:
                raise ValueError(
                    f"Sequence tensor must be 1D. Current shape is `{len(sequence.shape)}`"
                )
            padding = torch.as_tensor(padding)
        if pad_to_left:
            if isinstance(sequence, torch.Tensor):
                return torch.cat((padding, sequence), -1)
            return padding + sequence
        if isinstance(sequence, torch.Tensor):
            return torch.cat((sequence, padding), -1)
        return sequence + padding

    def convert_to_batch(
        self, samples: Any, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Convert the list of samples to a batch.

        Args:
            samples (:obj:`List`):
                List of samples to convert to a batch.

        Returns:
            :obj:`Dict[str, torch.Tensor]`: The batch.
        """
        # invert questions from list of dict to dict of list
        samples = {k: [d[k] for d in samples] for k in samples[0]}
        # get max length of questions
        max_len = max(len(x) for x in samples["input_ids"])
        # pad the questions
        for key in samples:
            if key in self.padding_ops:
                samples[key] = torch.as_tensor(
                    [self.padding_ops[key](b, max_len) for b in samples[key]]
                )
        return samples

    def shuffle_data(self, seed: int = 42):
        self.data = self.data.shuffle(seed=seed)


class InBatchNegativesDataset(GoldenRetrieverDataset):
    def __len__(self) -> int:
        if isinstance(self.data, datasets.Dataset):
            return len(self.data)

    def __getitem__(
        self, index
    ) -> Union[Dict[str, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        return self.data[index]

    def to_torch_dataset(self) -> torch.utils.data.Dataset:
        shuffle_this_time = self.shuffle

        if (
            self.subsample_strategy
            and self.subsample_strategy != SubsampleStrategyEnum.NONE
        ):
            number_of_samples = int(len(self.data) * self.subsample_portion)
            if self.subsample_strategy == SubsampleStrategyEnum.RANDOM:
                logger.info(
                    f"Random subsampling {number_of_samples} samples from {len(self.data)}"
                )
                data = (
                    deepcopy(self.data)
                    .shuffle(seed=42 + self.number_of_complete_iterations)
                    .select(range(0, number_of_samples))
                )
            elif self.subsample_strategy == SubsampleStrategyEnum.IN_ORDER:
                # number_of_samples = int(len(self.data) * self.subsample_portion)
                already_selected = (
                    number_of_samples * self.number_of_complete_iterations
                )
                logger.info(
                    f"Subsampling {number_of_samples} samples out of {len(self.data)}"
                )
                to_select = min(already_selected + number_of_samples, len(self.data))
                logger.info(
                    f"Portion of data selected: {already_selected} " f"to {to_select}"
                )
                data = deepcopy(self.data).select(range(already_selected, to_select))

                # don't shuffle the data if we are subsampling, and we have still not completed
                # one full iteration over the dataset
                if self.number_of_complete_iterations > 0:
                    shuffle_this_time = False

                # reset the number of complete iterations
                if to_select >= len(self.data):
                    # reset the number of complete iterations,
                    # we have completed one full iteration over the dataset
                    # the value is -1 because we want to start from 0 at the next iteration
                    self.number_of_complete_iterations = -1
            else:
                raise ValueError(
                    f"Subsample strategy `{self.subsample_strategy}` is not valid. "
                    f"Valid strategies are: {SubsampleStrategyEnum.__members__}"
                )

        else:
            data = data = self.data

        # do we need to shuffle the data?
        if self.shuffle and shuffle_this_time:
            logger.info("Shuffling the data")
            data = data.shuffle(seed=42 + self.number_of_complete_iterations)

        batch_fn_kwargs = {
            "passage_batch_size": self.passage_batch_size,
            "question_batch_size": self.question_batch_size,
            "hard_negatives_manager": self.hn_manager,
        }
        batched_data = self.create_batches(
            data,
            batch_fn=self.batch_fn,
            batch_fn_kwargs=batch_fn_kwargs,
            prefetch=self.prefetch,
        )

        batched_data = self.collate_batches(
            batched_data, self.collate_fn, prefetch=self.prefetch
        )

        # increment the number of complete iterations
        self.number_of_complete_iterations += 1

        if self.prefetch:
            return BaseDataset(name=self.name, data=batched_data)
        else:
            return IterableBaseDataset(name=self.name, data=batched_data)

    @staticmethod
    def load_fn(
        sample: Dict,
        tokenizer: tr.PreTrainedTokenizer,
        max_positives: int,
        max_negatives: int,
        max_hard_negatives: int,
        max_passages: int = -1,
        max_question_length: int = 256,
        max_passage_length: int = 128,
        *args,
        **kwargs,
    ) -> Dict:
        # remove duplicates and limit the number of passages
        positives = list(set([p["text"] for p in sample["positive_ctxs"]]))
        if max_positives != -1:
            positives = positives[:max_positives]
        negatives = list(set([n["text"] for n in sample["negative_ctxs"]]))
        if max_negatives != -1:
            negatives = negatives[:max_negatives]
        hard_negatives = list(set([h["text"] for h in sample["hard_negative_ctxs"]]))
        if max_hard_negatives != -1:
            hard_negatives = hard_negatives[:max_hard_negatives]

        question = tokenizer(
            sample["question"], max_length=max_question_length, truncation=True
        )

        passage = positives + negatives + hard_negatives
        if max_passages != -1:
            passage = passage[:max_passages]

        passage = tokenizer(passage, max_length=max_passage_length, truncation=True)

        # invert the passage data structure from a dict of lists to a list of dicts
        passage = [dict(zip(passage, t)) for t in zip(*passage.values())]

        output = dict(
            question=question,
            passage=passage,
            positives=positives,
            positive_pssgs=passage[: len(positives)],
        )
        return output

    @staticmethod
    def batch_fn(
        data: Dataset,
        passage_batch_size: int,
        question_batch_size: int,
        hard_negatives_manager: Optional[HardNegativesManager] = None,
        *args,
        **kwargs,
    ) -> Dict[str, List[Dict[str, Any]]]:
        def split_batch(
            batch: Union[Dict[str, Any], ModelInputs], question_batch_size: int
        ) -> List[ModelInputs]:
            """
            Split a batch into multiple batches of size `question_batch_size` while keeping
            the same number of passages.
            """

            def split_fn(x):
                return [
                    x[i : i + question_batch_size]
                    for i in range(0, len(x), question_batch_size)
                ]

            # split the sample_idx
            sample_idx = split_fn(batch["sample_idx"])
            # split the questions
            questions = split_fn(batch["questions"])
            # split the positives
            positives = split_fn(batch["positives"])
            # split the positives_pssgs
            positives_pssgs = split_fn(batch["positives_pssgs"])

            # collect the new batches
            batches = []
            for i in range(len(questions)):
                batches.append(
                    ModelInputs(
                        dict(
                            sample_idx=sample_idx[i],
                            questions=questions[i],
                            passages=batch["passages"],
                            positives=positives[i],
                            positives_pssgs=positives_pssgs[i],
                        )
                    )
                )
            return batches

        batch = []
        passages_in_batch = {}

        for sample in data:
            if len(passages_in_batch) >= passage_batch_size:
                # create the batch dict
                batch_dict = ModelInputs(
                    dict(
                        sample_idx=[s["sample_idx"] for s in batch],
                        questions=[s["question"] for s in batch],
                        passages=list(passages_in_batch.values()),
                        positives_pssgs=[s["positive_pssgs"] for s in batch],
                        positives=[s["positives"] for s in batch],
                    )
                )
                # split the batch if needed
                if len(batch) > question_batch_size:
                    for splited_batch in split_batch(batch_dict, question_batch_size):
                        yield splited_batch
                else:
                    yield batch_dict

                # reset batch
                batch = []
                passages_in_batch = {}

            batch.append(sample)
            # yes it's a bit ugly but it works :)
            # count the number of passages in the batch and stop if we reach the limit
            # we use a set to avoid counting the same passage twice
            # we use a tuple because set doesn't support lists
            # we use input_ids as discriminator
            passages_in_batch.update(
                {tuple(passage["input_ids"]): passage for passage in sample["passage"]}
            )
            # check for hard negatives and add with a probability of 0.1
            if hard_negatives_manager is not None:
                if sample["sample_idx"] in hard_negatives_manager:
                    passages_in_batch.update(
                        {
                            tuple(passage["input_ids"]): passage
                            for passage in hard_negatives_manager.get(
                                sample["sample_idx"]
                            )
                        }
                    )

        # left over
        if len(batch) > 0:
            # create the batch dict
            batch_dict = ModelInputs(
                dict(
                    sample_idx=[s["sample_idx"] for s in batch],
                    questions=[s["question"] for s in batch],
                    passages=list(passages_in_batch.values()),
                    positives_pssgs=[s["positive_pssgs"] for s in batch],
                    positives=[s["positives"] for s in batch],
                )
            )
            # split the batch if needed
            if len(batch) > question_batch_size:
                for splited_batch in split_batch(batch_dict, question_batch_size):
                    yield splited_batch
            else:
                yield batch_dict

    def collate_fn(self, batch: Any, *args, **kwargs) -> Any:
        # convert questions and passages to a batch
        questions = self.convert_to_batch(batch.questions)
        passages = self.convert_to_batch(batch.passages)

        # build an index to map the position of the passage in the batch
        passage_index = {tuple(c["input_ids"]): i for i, c in enumerate(batch.passages)}

        # now we can create the labels
        labels = torch.zeros(
            questions["input_ids"].shape[0], passages["input_ids"].shape[0]
        )
        # iterate over the questions and set the labels to 1 if the passage is positive
        for sample_idx in range(len(questions["input_ids"])):
            for pssg in batch["positives_pssgs"][sample_idx]:
                # get the index of the positive passage
                index = passage_index[tuple(pssg["input_ids"])]
                # set the label to 1
                labels[sample_idx, index] = 1

        model_inputs = ModelInputs(
            {
                "questions": questions,
                "passages": passages,
                "labels": labels,
                "positives": batch["positives"],
                "sample_idx": batch["sample_idx"],
            }
        )
        return model_inputs


class AidaInBatchNegativesDataset(InBatchNegativesDataset):
    def __init__(self, use_topics: bool = False, *args, **kwargs):
        if "load_fn_kwargs" not in kwargs:
            kwargs["load_fn_kwargs"] = {}
        kwargs["load_fn_kwargs"]["use_topics"] = use_topics
        super().__init__(*args, **kwargs)

    @staticmethod
    def load_fn(
        sample: Dict,
        tokenizer: tr.PreTrainedTokenizer,
        max_positives: int,
        max_negatives: int,
        max_hard_negatives: int,
        max_passages: int = -1,
        max_question_length: int = 256,
        max_passage_length: int = 128,
        use_topics: bool = False,
        *args,
        **kwargs,
    ) -> Dict:
        # remove duplicates and limit the number of passages
        positives = list(set([p["text"] for p in sample["positive_ctxs"]]))
        if max_positives != -1:
            positives = positives[:max_positives]
        negatives = list(set([n["text"] for n in sample["negative_ctxs"]]))
        if max_negatives != -1:
            negatives = negatives[:max_negatives]
        hard_negatives = list(set([h["text"] for h in sample["hard_negative_ctxs"]]))
        if max_hard_negatives != -1:
            hard_negatives = hard_negatives[:max_hard_negatives]

        question = sample["question"]

        if "doc_topic" in sample and use_topics:
            question = tokenizer(
                question,
                sample["doc_topic"],
                max_length=max_question_length,
                truncation=True,
            )
        else:
            question = tokenizer(
                question, max_length=max_question_length, truncation=True
            )

        passage = positives + negatives + hard_negatives
        if max_passages != -1:
            passage = passage[:max_passages]

        passage = tokenizer(passage, max_length=max_passage_length, truncation=True)

        # invert the passage data structure from a dict of lists to a list of dicts
        passage = [dict(zip(passage, t)) for t in zip(*passage.values())]

        output = dict(
            question=question,
            passage=passage,
            positives=positives,
            positive_pssgs=passage[: len(positives)],
        )
        return output
