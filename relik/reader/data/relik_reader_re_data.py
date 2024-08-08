import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
import tqdm
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from relik.reader.data.relik_reader_data_utils import (
    add_noise_to_value,
    batchify,
    batchify_matrices,
    batchify_tensor,
    chunks,
    flatten,
)
from relik.reader.data.relik_reader_sample import (
    RelikReaderSample,
    load_relik_reader_samples,
)
from relik.reader.utils.special_symbols import NME_SYMBOL

from torch.utils.data import get_worker_info
import torch.distributed as dist

logger = logging.getLogger(__name__)


class TokenizationOutput(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    prediction_mask: torch.Tensor
    special_symbols_mask: torch.Tensor
    special_symbols_mask_entities: torch.Tensor


class RelikREDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: Optional[str],
        materialize_samples: bool,
        transformer_model: Union[str, PreTrainedTokenizer],
        special_symbols_re: List[str],
        special_symbols: Optional[List[str]] = [],
        shuffle_candidates: Optional[Union[bool, float]] = False,
        flip_candidates: Optional[Union[bool, float]] = False,
        for_inference: bool = False,
        noise_param: float = 0.1,
        sorting_fields: Optional[str] = None,
        tokens_per_batch: int = 2048,
        batch_size: int = None,
        max_batch_size: int = 128,
        section_size: int = 500_000,
        prebatch: bool = True,
        add_gold_candidates: bool = True,
        use_nme: bool = False,
        min_length: int = -1,
        max_length: int = 2048,
        max_triplets: int = 50,
        max_spans: int = 100,
        model_max_length: int = 2048,
        skip_empty_training_samples: bool = True,
        drop_last: bool = False,
        samples: Optional[Iterator[RelikReaderSample]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.materialize_samples = materialize_samples
        self.samples: Optional[List[RelikReaderSample]] = samples
        if self.materialize_samples and self.samples is None:
            self.samples = list()

        self.special_symbols = special_symbols_re
        self.special_symbols_types = special_symbols

        if isinstance(transformer_model, str):
            self.tokenizer = self._build_tokenizer(
                transformer_model, self.special_symbols + self.special_symbols_types
            )
        else:
            self.tokenizer = transformer_model

        self.shuffle_candidates = shuffle_candidates
        self.flip_candidates = flip_candidates
        self.for_inference = for_inference
        self.noise_param = noise_param
        self.batching_fields = ["input_ids"]
        self.sorting_fields = (
            sorting_fields if sorting_fields is not None else self.batching_fields
        )
        self.add_gold_candidates = add_gold_candidates
        self.use_nme = use_nme
        self.min_length = min_length
        self.max_length = max_length
        self.model_max_length = (
            model_max_length
            if model_max_length < self.tokenizer.model_max_length
            else self.tokenizer.model_max_length
        )
        self.transformer_model = transformer_model
        self.skip_empty_training_samples = skip_empty_training_samples
        self.drop_last = drop_last

        self.tokens_per_batch = tokens_per_batch
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.max_triplets = max_triplets
        self.max_spans = max_spans
        self.section_size = section_size
        self.prebatch = prebatch

    def _build_tokenizer(self, transformer_model: str, special_symbols: List[str]):
        return AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=[ss for ss in special_symbols],
            add_prefix_space=True,
        )

    @staticmethod
    def get_special_symbols_re(num_entities: int, use_nme: bool = False) -> List[str]:
        if use_nme:
            return [NME_SYMBOL] + [f"[R-{i}]" for i in range(num_entities)]
        else:
            return [f"[R-{i}]" for i in range(num_entities)]

    @staticmethod
    def get_special_symbols(num_entities: int) -> List[str]:
        return [NME_SYMBOL] + [f"[E-{i}]" for i in range(num_entities)]

    @property
    def fields_batcher(self) -> Dict[str, Union[None, Callable[[list], Any]]]:
        fields_batchers = {
            "input_ids": lambda x: batchify(
                x, padding_value=self.tokenizer.pad_token_id
            ),
            "attention_mask": lambda x: batchify(x, padding_value=0),
            "token_type_ids": lambda x: batchify(x, padding_value=0),
            "prediction_mask": lambda x: batchify(x, padding_value=1),
            "global_attention": lambda x: batchify(x, padding_value=0),
            "token2word": None,
            "sample": None,
            "special_symbols_mask": lambda x: batchify(x, padding_value=False),
            "special_symbols_mask_entities": lambda x: batchify(x, padding_value=False),
            "start_labels": lambda x: batchify(x, padding_value=-100),
            "end_labels": lambda x: batchify_matrices(x, padding_value=-100),
            "disambiguation_labels": lambda x: batchify(x, padding_value=-100),
            "relation_labels": lambda x: batchify_tensor(x, padding_value=-100),
            "predictable_candidates": None,
        }
        if (
            isinstance(self.transformer_model, str)
            and "roberta" in self.transformer_model
        ) or (
            isinstance(self.transformer_model, PreTrainedTokenizer)
            and "roberta" in self.transformer_model.config.model_type
        ):
            del fields_batchers["token_type_ids"]

        return fields_batchers

    def _build_input_ids(
        self, sentence_input_ids: List[int], candidates_input_ids: List[List[int]]
    ) -> List[int]:
        return (
            [self.tokenizer.cls_token_id]
            + sentence_input_ids
            + [self.tokenizer.sep_token_id]
            + flatten(candidates_input_ids)
            + [self.tokenizer.sep_token_id]
        )

    def _build_input(self, text: List[str], candidates: List[List[str]]) -> List[int]:
        return (
            text
            + [self.tokenizer.sep_token]
            + flatten(candidates)
            + [self.tokenizer.sep_token]
        )

    def _build_tokenizer_essentials(
        self, input_ids, original_sequence, ents=0
    ) -> TokenizationOutput:
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        if len(self.special_symbols_types) > 0:
            # special symbols mask
            special_symbols_mask = input_ids >= self.tokenizer.vocab_size
            # select only the first N true values where N is len(entities_definitions)
            special_symbols_mask_entities = special_symbols_mask.clone()
            special_symbols_mask_entities[
                special_symbols_mask_entities.cumsum(0) > ents
            ] = False
            token_type_ids = (torch.cumsum(special_symbols_mask, dim=0) > 0).long()
            special_symbols_mask = special_symbols_mask ^ special_symbols_mask_entities
        else:
            special_symbols_mask = input_ids >= self.tokenizer.vocab_size
            special_symbols_mask_entities = special_symbols_mask.clone()
            token_type_ids = (torch.cumsum(special_symbols_mask, dim=0) > 0).long()

        prediction_mask = token_type_ids.roll(shifts=-1, dims=0)
        prediction_mask[-1] = 1
        prediction_mask[0] = 1

        assert len(prediction_mask) == len(input_ids)

        return TokenizationOutput(
            input_ids,
            attention_mask,
            token_type_ids,
            prediction_mask,
            special_symbols_mask,
            special_symbols_mask_entities,
        )

    @staticmethod
    def _subindex(lst, target_values, dims):
        for i, sublist in enumerate(lst):
            match = all(sublist[dim] == target_values[dim] for dim in dims)
            if match:
                return i

    def _build_labels(
        self,
        sample,
        tokenization_output: TokenizationOutput,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        len_input_ids = len(tokenization_output.input_ids)
        start_labels = [0] * len_input_ids
        end_labels = []
        end_labels_tensor = [0] * len_input_ids

        # Pre-process entities to handle edge cases and duplicates.
        sample.window_labels_tokens.sort(key=lambda x: (x[0], x[1]))
        processed_entities = []
        entities_untyped = set()
        for entity in sample.window_labels_tokens:
            entity = self._adjust_entity_boundaries(entity, sample.word2token)
            if entity:
                processed_entities.append(entity)
                entities_untyped.add((entity[0], entity[1]))

        # Initialize disambiguation and relation labels.
        entities_untyped = sorted(entities_untyped)
        num_disambiguation_labels = (
            len(sample.span_candidates) + len(sample.triplet_candidates)
            if self.special_symbols_types
            else len(sample.triplet_candidates)
        )
        disambiguation_labels = torch.zeros(
            (len(entities_untyped), num_disambiguation_labels)
        )

        # Main loop to fill in start and end labels.
        prev_start_bpe = -1
        offset = 0
        for idx, entity in enumerate(processed_entities):
            start_bpe, end_bpe = self._get_bpe_positions(entity, sample.word2token)
            class_index = idx + 1  # Offset by 1 for NONE class

            # Update start labels
            start_labels[start_bpe] = class_index

            # Handle end labels, considering overlapping entities.
            if start_bpe != prev_start_bpe:
                end_labels.append(end_labels_tensor.copy())
                end_labels[-1][:start_bpe] = [-100] * start_bpe
            elif end_labels[-1][end_bpe] != 0:
                # Handle overlapping entities by adjusting the start position.
                offset += 1
                prev_start_bpe = start_bpe
                continue
            end_labels[-1][end_bpe] = class_index

            # Update disambiguation labels if applicable.
            if self.special_symbols_types and entity[2] in sample.span_candidates:
                disambiguation_labels[
                    idx - offset, sample.span_candidates.index(entity[2])
                ] = 1
            prev_start_bpe = start_bpe

        # Handle ignored indices for start and end labels.
        ignored_indices = tokenization_output.prediction_mask == 1
        start_labels_tensor = torch.tensor(start_labels, dtype=torch.long)
        start_labels_tensor[ignored_indices] = -100
        end_labels_tensor = torch.stack([torch.tensor(label) for label in end_labels])
        end_labels_tensor[ignored_indices.repeat(len(end_labels), 1)] = -100

        # Initialize and update relation labels.
        relation_labels = self._init_relation_labels(entities_untyped, sample)

        return (
            start_labels_tensor,
            end_labels_tensor,
            disambiguation_labels,
            relation_labels,
        )

    def _adjust_entity_boundaries(self, entity, word2token):
        """Adjust entity boundaries to ensure they map to valid token positions."""
        # Adjust start position
        while not word2token.get(entity[0], []):
            entity = (entity[0] + 1, entity[1], entity[2])
            if entity[0] >= len(word2token):
                return None
        # Adjust end position
        while not word2token.get(entity[1] - 1, []):
            entity = (entity[0], entity[1] - 1, entity[2])
            if entity[1] <= 0:
                return None
        return entity

    def _get_bpe_positions(self, entity, word2token):
        """Get the start and end positions in BPE tokens for an entity."""
        return word2token[entity[0]][0] + 1, word2token[entity[1] - 1][-1] + 1

    def _init_relation_labels(self, entities_untyped, sample):
        """
        Initialize the tensor for relation labels between entities.
        """
        num_entities = len(entities_untyped)
        num_relations = len(sample.triplet_candidates)
        relation_labels = torch.zeros((num_entities, num_entities, num_relations))

        if sample.window_triplet_labels_tokens is None:
            relation_labels.fill_(-100)
            return relation_labels

        for relation in sample.window_triplet_labels_tokens:
            # Determine the index of this relation type in the triplet_candidates list.
            # If the relation type is not found, use a default index (e.g., for 'unknown' relation types).
            relation_idx = (
                sample.triplet_candidates.index(relation["relation"])
                if relation["relation"] in sample.triplet_candidates
                else -1
            )

            # Find the indices of the subject and object entities within the entities_untyped list.
            subject_idx = next(
                (
                    i
                    for i, entity in enumerate(entities_untyped)
                    if entity[0] == relation["subject"][0]
                    and entity[1] == relation["subject"][1]
                ),
                None,
            )
            object_idx = next(
                (
                    i
                    for i, entity in enumerate(entities_untyped)
                    if entity[0] == relation["object"][0]
                    and entity[1] == relation["object"][1]
                ),
                None,
            )

            # If both the subject and object entity indices are found, mark the relation in the tensor.
            if (
                subject_idx is not None
                and object_idx is not None
                and relation_idx != -1
            ):
                relation_labels[subject_idx, object_idx, relation_idx] = 1

        return relation_labels

    def __iter__(self):
        dataset_iterator = self.dataset_iterator_func()
        i = None
        if self.section_size is not None:
            current_dataset_elements = []
            for i, dataset_elem in enumerate(dataset_iterator, start=1):
                if len(current_dataset_elements) == self.section_size:
                    for batch in self.materialize_batches(current_dataset_elements):
                        yield batch
                    current_dataset_elements = []
                current_dataset_elements.append(dataset_elem)
                if i % 50_000 == 0:
                    logger.info(f"Processed: {i} number of elements")
            if len(current_dataset_elements) != 0:
                for batch in self.materialize_batches(current_dataset_elements):
                    yield batch
            if i is not None:
                logger.debug(f"Dataset finished: {i} number of elements processed")
            else:
                logger.warning("Dataset empty")
        else:
            for batch in self.materialize_batches(dataset_iterator):
                if i is None:
                    i = 0
                i += batch["input_ids"].shape[0]
                yield batch
            if i is not None:
                logger.debug(f"Dataset finished: {i} number of elements processed")
            else:
                logger.warning("Dataset empty")

    def dataset_iterator_func(self):
        data_samples = (
            load_relik_reader_samples(self.dataset_path)
            if self.samples is None
            or (isinstance(self.samples, list) and len(self.samples) == 0)
            else self.samples
        )
        if self.materialize_samples:
            data_acc = []
        # take care of the tqdm nesting
        # for sample in tqdm.tqdm(data_samples, desc="Reading dataset"):

        # Handle DDP
        world_size = 1
        rank = 0
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()

        def partition_data_samples(iterable, world_size, rank):
            def generator():
                for i, x in enumerate(iterable):
                    if i % world_size == rank:
                        yield x

            return generator

        samples_per_rank = partition_data_samples(data_samples, world_size, rank)()

        for sample in samples_per_rank:
            if self.materialize_samples and sample.materialize is not None:
                # tokenization_output = sample.materialize["tokenization_output"]
                materialized = sample.materialize
                del sample.materialize
                yield {
                    "input_ids": materialized["tokenization_output"].input_ids,
                    "attention_mask": materialized[
                        "tokenization_output"
                    ].attention_mask,
                    "token_type_ids": materialized[
                        "tokenization_output"
                    ].token_type_ids,
                    "prediction_mask": materialized[
                        "tokenization_output"
                    ].prediction_mask,
                    "special_symbols_mask": materialized[
                        "tokenization_output"
                    ].special_symbols_mask,
                    "special_symbols_mask_entities": materialized[
                        "tokenization_output"
                    ].special_symbols_mask_entities,
                    "sample": sample,
                    "start_labels": materialized["start_labels"],
                    "end_labels": materialized["end_labels"],
                    "disambiguation_labels": materialized["disambiguation_labels"],
                    "relation_labels": materialized["relation_labels"],
                    "predictable_candidates": materialized["candidates_symbols"],
                }
                sample.materialize = materialized
                data_acc.append(sample)
                continue
            candidates_symbols = self.special_symbols
            candidates_entities_symbols = self.special_symbols_types

            # sample.candidates = sample.candidates[: self.max_candidates]

            if len(self.special_symbols_types) > 0:
                # sample.span_candidates = sample.span_candidates[
                #     : self.max_ent_candidates
                # ]
                # add NME as a possible candidate
                assert sample.span_candidates is not None
                if self.use_nme:
                    sample.span_candidates.insert(0, NME_SYMBOL)
                    # sample.candidates.insert(0, NME_SYMBOL)

            sample.triplet_candidates = sample.triplet_candidates[
                : min(len(candidates_symbols), self.max_triplets)
            ]

            if len(self.special_symbols_types) > 0:
                sample.span_candidates = sample.span_candidates[
                    : min(len(candidates_entities_symbols), self.max_spans)
                ]
            # training time sample mods
            if not self.for_inference:
                # check whether the sample has labels if not skip
                if (
                    sample.window_triplet_labels_tokens is None
                    or len(sample.window_triplet_labels_tokens) == 0
                ) and self.skip_empty_training_samples:
                    # logger.warning(
                    #     "Sample {} has no labels, skipping".format(sample.doc_id)
                    # )
                    continue

                # add gold candidates if missing
                if self.add_gold_candidates and sample.window_triplet_labels_tokens:
                    candidates_set = set(sample.triplet_candidates)
                    candidates_to_add = set()
                    for candidate_title in sample.window_triplet_labels_tokens:
                        if candidate_title["relation"] not in candidates_set:
                            candidates_to_add.add(candidate_title["relation"])
                    if len(candidates_to_add) > 0:
                        # replacing last candidates with the gold ones
                        # this is done in order to preserve the ordering
                        candidates_to_add = list(candidates_to_add)
                        added_gold_candidates = 0
                        gold_candidates_titles_set = set(
                            set(
                                ct["relation"]
                                for ct in sample.window_triplet_labels_tokens
                            )
                        )
                        for i in reversed(range(len(sample.triplet_candidates))):
                            if (
                                sample.triplet_candidates[i]
                                not in gold_candidates_titles_set
                                and sample.triplet_candidates[i] != NME_SYMBOL
                            ):
                                sample.triplet_candidates[i] = candidates_to_add[
                                    added_gold_candidates
                                ]
                                added_gold_candidates += 1
                                if len(candidates_to_add) == added_gold_candidates:
                                    break

                        candidates_still_to_add = (
                            len(candidates_to_add) - added_gold_candidates
                        )
                        while (
                            len(sample.triplet_candidates)
                            <= min(len(candidates_symbols), self.max_triplets)
                            and candidates_still_to_add != 0
                        ):
                            sample.triplet_candidates.append(
                                candidates_to_add[added_gold_candidates]
                            )
                            added_gold_candidates += 1
                            candidates_still_to_add -= 1

                def shuffle_cands(shuffle_candidates, candidates):
                    if (
                        isinstance(shuffle_candidates, bool) and shuffle_candidates
                    ) or (
                        isinstance(shuffle_candidates, float)
                        and np.random.uniform() < shuffle_candidates
                    ):
                        np.random.shuffle(candidates)
                        if NME_SYMBOL in candidates:
                            candidates.remove(NME_SYMBOL)
                            candidates.insert(0, NME_SYMBOL)
                    return candidates

                def flip_cands(flip_candidates, candidates):
                    # flip candidates
                    if (isinstance(flip_candidates, bool) and flip_candidates) or (
                        isinstance(flip_candidates, float)
                        and np.random.uniform() < flip_candidates
                    ):
                        for i in range(len(candidates) - 1):
                            if np.random.uniform() < 0.5:
                                candidates[i], candidates[i + 1] = (
                                    candidates[i + 1],
                                    candidates[i],
                                )
                        if NME_SYMBOL in candidates:
                            candidates.remove(NME_SYMBOL)
                            candidates.insert(0, NME_SYMBOL)
                    return candidates

                if self.shuffle_candidates:
                    sample.triplet_candidates = shuffle_cands(
                        self.shuffle_candidates, sample.triplet_candidates
                    )
                    if len(self.special_symbols_types) > 0:
                        sample.span_candidates = shuffle_cands(
                            self.shuffle_candidates, sample.span_candidates
                        )
                elif self.flip_candidates:
                    sample.triplet_candidates = flip_cands(
                        self.flip_candidates, sample.triplet_candidates
                    )
                    if len(self.special_symbols_types) > 0:
                        sample.span_candidates = flip_cands(
                            self.flip_candidates, sample.span_candidates
                        )

            # candidates encoding
            candidates_symbols = candidates_symbols[: len(sample.triplet_candidates)]

            candidates_encoding = [
                ["{} {}".format(cs, ct)] if ct != NME_SYMBOL else [NME_SYMBOL]
                for cs, ct in zip(candidates_symbols, sample.triplet_candidates)
            ]
            if len(self.special_symbols_types) > 0:
                candidates_entities_symbols = candidates_entities_symbols[
                    : len(sample.span_candidates)
                ]
                candidates_types_encoding = [
                    ["{} {}".format(cs, ct)] if ct != NME_SYMBOL else [NME_SYMBOL]
                    for cs, ct in zip(
                        candidates_entities_symbols, sample.span_candidates
                    )
                ]
                candidates_encoding = (
                    candidates_types_encoding
                    + [[self.tokenizer.sep_token]]
                    + candidates_encoding
                )

            pretoken_input = self._build_input(sample.words, candidates_encoding)
            input_tokenized = self.tokenizer(
                pretoken_input,
                return_offsets_mapping=True,
                add_special_tokens=False,
            )

            window_tokens = input_tokenized.input_ids
            window_tokens = flatten(window_tokens)

            offsets_mapping = [
                [
                    (
                        ss + sample.token2char_start[str(i)],
                        se + sample.token2char_start[str(i)],
                    )
                    for ss, se in input_tokenized.offset_mapping[i]
                ]
                for i in range(len(sample.words))
            ]

            offsets_mapping = flatten(offsets_mapping)

            token2char_start = {str(i): s for i, (s, _) in enumerate(offsets_mapping)}
            token2char_end = {str(i): e for i, (_, e) in enumerate(offsets_mapping)}
            token2word_start = {
                str(i): int(sample._d["char2token_start"][str(s)])
                for i, (s, _) in enumerate(offsets_mapping)
                if str(s) in sample._d["char2token_start"]
            }
            token2word_end = {
                str(i): int(sample._d["char2token_end"][str(e)])
                for i, (_, e) in enumerate(offsets_mapping)
                if str(e) in sample._d["char2token_end"]
            }
            # invert token2word_start and token2word_end
            word2token_start = {str(v): int(k) for k, v in token2word_start.items()}
            word2token_end = {str(v): int(k) for k, v in token2word_end.items()}

            sample._d.update(
                dict(
                    tokens=window_tokens,
                    token2char_start=token2char_start,
                    token2char_end=token2char_end,
                    token2word_start=token2word_start,
                    token2word_end=token2word_end,
                    word2token_start=word2token_start,
                    word2token_end=word2token_end,
                )
            )

            input_subwords = flatten(input_tokenized["input_ids"][: len(sample.words)])
            offsets = input_tokenized["offset_mapping"][: len(sample.words)]
            token2word = []
            word2token = {}
            count = 0
            for i, offset in enumerate(offsets):
                word2token[i] = []
                for token in offset:
                    token2word.append(i)
                    word2token[i].append(count)
                    count += 1

            sample.token2word = token2word
            sample.word2token = word2token
            candidates_encoding_result = input_tokenized["input_ids"][
                len(sample.words) + 1 : -1
            ]

            i = 0
            cum_len = 0
            # drop candidates if the number of input tokens is too long for the model
            if (
                sum(map(len, candidates_encoding_result))
                + len(input_subwords)
                + 20  # + 20 special tokens
                > self.model_max_length
            ):
                if self.for_inference:
                    acceptable_tokens_from_candidates = (
                        self.model_max_length - 20 - len(input_subwords)
                    )
                    current_len = (
                        len(candidates_encoding_result[i])
                        + len(
                            candidates_encoding_result[i + len(sample.span_candidates)]
                        )
                        if len(self.special_symbols_types) > 0
                        else len(candidates_encoding_result[i])
                    )
                    while cum_len + current_len < acceptable_tokens_from_candidates:
                        cum_len += current_len
                        i += 1
                        if len(self.special_symbols_types) == 0 or i + len(
                            sample.span_candidates
                        ) >= len(candidates_encoding_result):
                            current_len = len(candidates_encoding_result[i])
                        else:
                            current_len = len(
                                candidates_encoding_result[i]
                                + candidates_encoding_result[
                                    i + len(sample.span_candidates)
                                ]
                            )

                    assert i > 0

                    candidates_encoding_result = (
                        candidates_encoding_result[:i]
                        + candidates_encoding_result[len(sample.span_candidates) : i]
                    )
                    if len(self.special_symbols_types) > 0:
                        candidates_entities_symbols = candidates_entities_symbols[:i]
                        sample.span_candidates = sample.span_candidates[:i]

                    candidates_symbols = candidates_symbols[:i]
                    sample.triplet_candidates = sample.triplet_candidates[:i]
                else:
                    if len(sample.window_triplet_labels_tokens) == 0:
                        sample.window_triplet_labels_tokens = []
                    gold_candidates_set = set(
                        [wl["relation"] for wl in sample.window_triplet_labels_tokens]
                    )
                    gold_candidates_indices = [
                        i
                        for i, wc in enumerate(sample.triplet_candidates)
                        if wc in gold_candidates_set
                    ]
                    if len(self.special_symbols_types) > 0:
                        gold_candidates_indices = [
                            i + len(sample.span_candidates)
                            for i in gold_candidates_indices
                        ] + [len(sample.span_candidates)]
                        gold_candidates_set_entities = set(
                            [wl[2] for wl in sample.window_labels_tokens]
                        )
                        gold_candidates_indices += [
                            i
                            for i, wc in enumerate(sample.span_candidates)
                            if wc in gold_candidates_set_entities
                        ]
                        # # add entities indices
                        # gold_candidates_indices = gold_candidates_indices + list(
                        #     range(len(sample.span_candidates))
                        # )
                    necessary_taken_tokens = sum(
                        map(
                            len,
                            [
                                candidates_encoding_result[i]
                                for i in gold_candidates_indices
                            ],
                        )
                    )

                    acceptable_tokens_from_candidates = (
                        self.model_max_length
                        - 20
                        - len(input_subwords)
                        - necessary_taken_tokens
                    )
                    if acceptable_tokens_from_candidates <= 0:
                        logger.warning(
                            "Sample {} has no candidates after truncation due to max length".format(
                                sample.doc_id
                            )
                        )
                        continue
                    # assert acceptable_tokens_from_candidates > 0

                    i = 0
                    cum_len = 0
                    while (
                        cum_len + len(candidates_encoding_result[i])
                        < acceptable_tokens_from_candidates
                    ):
                        if i not in gold_candidates_indices:
                            cum_len += len(candidates_encoding_result[i])
                        i += 1

                    new_indices = sorted(
                        list(set(list(range(i)) + gold_candidates_indices))
                    )
                    # np.random.shuffle(new_indices)

                    candidates_encoding_result = [
                        candidates_encoding_result[i] for i in new_indices
                    ]
                    if len(self.special_symbols_types) > 0:
                        sample.triplet_candidates = [
                            sample.triplet_candidates[i - len(sample.span_candidates)]
                            for i in new_indices[len(sample.span_candidates) : -1]
                        ]
                        candidates_symbols = candidates_symbols[
                            : len(sample.triplet_candidates)
                        ]
                    else:
                        candidates_symbols = [
                            candidates_symbols[i] for i in new_indices
                        ]
                        sample.triplet_candidates = [
                            sample.triplet_candidates[i] for i in new_indices
                        ]
                if len(sample.triplet_candidates) == 0:
                    logger.warning(
                        "Sample {} has no candidates after truncation due to max length".format(
                            sample.doc_id
                        )
                    )
                    continue

            # final input_ids build
            input_ids = self._build_input_ids(
                sentence_input_ids=input_subwords,
                candidates_input_ids=candidates_encoding_result,
            )

            # complete input building (e.g. attention / prediction mask)
            tokenization_output = self._build_tokenizer_essentials(
                input_ids,
                input_subwords,
                (
                    min(len(sample.span_candidates), len(self.special_symbols_types))
                    if sample.span_candidates is not None
                    else 0
                ),
            )
            # labels creation
            start_labels, end_labels, disambiguation_labels, relation_labels = (
                None,
                None,
                None,
                None,
            )
            if (
                sample.window_labels_tokens is not None
                and len(sample.window_labels_tokens) > 0
            ):
                (
                    start_labels,
                    end_labels,
                    disambiguation_labels,
                    relation_labels,
                ) = self._build_labels(
                    sample,
                    tokenization_output,
                )
            elif not self.for_inference:
                continue

            if self.materialize_samples:
                sample.materialize = {
                    "tokenization_output": tokenization_output,
                    "start_labels": start_labels,
                    "end_labels": end_labels,
                    "disambiguation_labels": disambiguation_labels,
                    "relation_labels": relation_labels,
                    "candidates_symbols": candidates_symbols,
                }
                data_acc.append(sample)
            yield {
                "input_ids": tokenization_output.input_ids,
                "attention_mask": tokenization_output.attention_mask,
                "token_type_ids": tokenization_output.token_type_ids,
                "prediction_mask": tokenization_output.prediction_mask,
                "special_symbols_mask": tokenization_output.special_symbols_mask,
                "special_symbols_mask_entities": tokenization_output.special_symbols_mask_entities,
                "sample": sample,
                "start_labels": start_labels,
                "end_labels": end_labels,
                "disambiguation_labels": disambiguation_labels,
                "relation_labels": relation_labels,
                "predictable_candidates": candidates_symbols
                + candidates_entities_symbols,
            }
        if self.materialize_samples:
            self.samples = data_acc

    def preshuffle_elements(self, dataset_elements: List):
        # This shuffling is done so that when using the sorting function,
        # if it is deterministic given a collection and its order, we will
        # make the whole operation not deterministic anymore.
        # Basically, the aim is not to build every time the same batches.
        if not self.for_inference:
            dataset_elements = np.random.permutation(dataset_elements)

        sorting_fn = lambda elem: (
            add_noise_to_value(
                sum(len(elem[k]) for k in self.sorting_fields),
                noise_param=self.noise_param,
            )
            if not self.for_inference
            else sum(len(elem[k]) for k in self.sorting_fields)
        )

        dataset_elements = sorted(dataset_elements, key=sorting_fn)

        if self.for_inference:
            return dataset_elements

        ds = list(chunks(dataset_elements, 64))  # todo: modified
        np.random.shuffle(ds)
        return flatten(ds)

    def materialize_batches(
        self, dataset_elements: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        if self.prebatch and self.section_size is not None and not self.for_inference:
            dataset_elements = self.preshuffle_elements(dataset_elements)

        current_batch = []

        # function that creates a batch from the 'current_batch' list
        def output_batch() -> Dict[str, Any]:
            assert (
                len(
                    set([len(elem["predictable_candidates"]) for elem in current_batch])
                )
                == 1
            ), " ".join(
                map(
                    str, [len(elem["predictable_candidates"]) for elem in current_batch]
                )
            )

            batch_dict = dict()

            de_values_by_field = {
                fn: [de[fn] for de in current_batch if fn in de]
                for fn in self.fields_batcher
            }

            # in case you provide fields batchers but in the batch
            # there are no elements for that field
            de_values_by_field = {
                fn: fvs for fn, fvs in de_values_by_field.items() if len(fvs) > 0
            }

            assert len(set([len(v) for v in de_values_by_field.values()]))

            # todo: maybe we should report the user about possible
            #  fields filtering due to "None" instances
            de_values_by_field = {
                fn: fvs
                for fn, fvs in de_values_by_field.items()
                if all([fv is not None for fv in fvs])
            }

            for field_name, field_values in de_values_by_field.items():
                field_batch = (
                    self.fields_batcher[field_name](field_values)
                    if self.fields_batcher[field_name] is not None
                    else field_values
                )

                batch_dict[field_name] = field_batch

            return batch_dict

        max_len_discards, min_len_discards = 0, 0

        should_token_batch = self.batch_size is None

        curr_pred_elements = -1
        for de in dataset_elements:
            if (
                should_token_batch
                and self.max_batch_size != -1
                and len(current_batch) == self.max_batch_size
            ) or (not should_token_batch and len(current_batch) == self.batch_size):
                yield output_batch()
                current_batch = []
                curr_pred_elements = -1

            # todo support max length (and min length) as dicts

            too_long_fields = [
                k
                for k in de
                if self.max_length != -1
                and torch.is_tensor(de[k])
                and len(de[k]) > self.max_length
            ]
            if len(too_long_fields) > 0:
                max_len_discards += 1
                continue

            too_short_fields = [
                k
                for k in de
                if self.min_length != -1
                and torch.is_tensor(de[k])
                and len(de[k]) < self.min_length
            ]
            if len(too_short_fields) > 0:
                min_len_discards += 1
                continue

            if should_token_batch:
                de_len = sum(len(de[k]) for k in self.batching_fields)

                future_max_len = max(
                    de_len,
                    max(
                        [
                            sum(len(bde[k]) for k in self.batching_fields)
                            for bde in current_batch
                        ],
                        default=0,
                    ),
                )

                future_tokens_per_batch = future_max_len * (len(current_batch) + 1)

                num_predictable_candidates = len(de["predictable_candidates"])

                if len(current_batch) > 0 and (
                    future_tokens_per_batch >= self.tokens_per_batch
                    or (
                        num_predictable_candidates != curr_pred_elements
                        and curr_pred_elements != -1
                    )
                ):
                    yield output_batch()
                    current_batch = []

            current_batch.append(de)
            curr_pred_elements = len(de["predictable_candidates"])

        if len(current_batch) != 0 and not self.drop_last:
            yield output_batch()

        if max_len_discards > 0:
            if self.for_inference:
                logger.warning(
                    f"WARNING: Inference mode is True but {max_len_discards} samples longer than max length were "
                    f"found. The {max_len_discards} samples will be DISCARDED. If you are doing some kind of evaluation"
                    f", this can INVALIDATE results. This might happen if the max length was not set to -1 or if the "
                    f"sample length exceeds the maximum length supported by the current model."
                )
            else:
                logger.warning(
                    f"During iteration, {max_len_discards} elements were "
                    f"discarded since longer than max length {self.max_length}"
                )

        if min_len_discards > 0:
            if self.for_inference:
                logger.warning(
                    f"WARNING: Inference mode is True but {min_len_discards} samples shorter than min length were "
                    f"found. The {min_len_discards} samples will be DISCARDED. If you are doing some kind of evaluation"
                    f", this can INVALIDATE results. This might happen if the min length was not set to -1 or if the "
                    f"sample length is shorter than the minimum length supported by the current model."
                )
            else:
                logger.warning(
                    f"During iteration, {min_len_discards} elements were "
                    f"discarded since shorter than min length {self.min_length}"
                )

    @staticmethod
    def _new_output_format(sample: RelikReaderSample) -> RelikReaderSample:
        # try-out for a new format

        # set of span tuples (start, end, type) for each entity
        predicted_spans = set()
        for prediction in sample.predicted_entities:
            predicted_spans.add(
                (
                    prediction[0],
                    prediction[1],
                    prediction[2],
                )
            )

        # sort the spans by start so that we can use the index of the span to get the entity
        predicted_spans = sorted(predicted_spans, key=lambda x: x[0])
        predicted_triplets = []
        # now search for the spans in each triplet
        for prediction in sample.predicted_relations:
            # get the index of the entity that has the same start and end
            start_entity_index = [
                i
                for i, p in enumerate(predicted_spans)
                if p[:2] == (prediction["subject"][0], prediction["subject"][1])
            ][0]
            end_entity_index = [
                i
                for i, p in enumerate(predicted_spans)
                if p[:2] == (prediction["object"][0], prediction["object"][1])
            ][0]

            predicted_triplets.append(
                (
                    start_entity_index,
                    prediction["relation"],
                    end_entity_index,
                    prediction["probability"],
                )
            )
        sample.predicted_spans = predicted_spans
        sample.predicted_triplets = predicted_triplets
        return sample

    @staticmethod
    def _convert_annotations(sample: RelikReaderSample) -> RelikReaderSample:
        triplets = []
        entities = []

        for entity in sample.predicted_entities:
            span_start = entity[0] - 1
            span_end = entity[1] - 1
            if str(span_start) not in sample.token2word_start:
                # span_start is in the middle of a word
                # retrieve the first token of the word
                while str(span_start) not in sample.token2word_start:
                    span_start -= 1
                    # skip
                    if span_start < 0:
                        break
            if str(span_end) not in sample.token2word_end:
                # span_end is in the middle of a word
                # retrieve the last token of the word
                while str(span_end) not in sample.token2word_end:
                    span_end += 1
                    # skip
                    if span_end >= len(sample.tokens):
                        break

            if span_start < 0 or span_end >= len(sample.tokens):
                continue

            entities.append(
                (
                    sample.token2word_start[str(span_start)],
                    sample.token2word_end[str(span_end)] + 1,
                    (
                        sample.span_candidates[entity[2]]
                        if sample.span_candidates and len(entity) > 2
                        else "--NME--"
                    ),
                )
            )
        for predicted_triplet, predicted_triplet_probabilities in zip(
            sample.predicted_relations, sample.predicted_relations_probabilities
        ):
            subject, object_, relation = predicted_triplet
            subject = entities[subject]
            object_ = entities[object_]
            relation = sample.triplet_candidates[relation]
            triplets.append(
                {
                    "subject": subject,
                    "relation": relation,
                    "object": object_,
                    "probability": float(predicted_triplet_probabilities.round(2)),
                }
            )
        # convert to list since we need to modify the sample down the road
        sample.predicted_entities = entities
        sample.predicted_relations = triplets
        del sample._d["predicted_relations_probabilities"]

        return sample

    @staticmethod
    def convert_to_word_annotations(sample: RelikReaderSample) -> RelikReaderSample:
        sample = RelikREDataset._convert_annotations(sample)
        return RelikREDataset._new_output_format(sample)

    @staticmethod
    def convert_to_char_annotations(
        sample: RelikReaderSample,
        remove_nmes: bool = True,
    ) -> RelikReaderSample:
        RelikREDataset._convert_annotations(sample)
        if "token2char_start" in sample._d:
            entities = []
            for entity in sample.predicted_entities:
                entity = list(entity)
                token_start = sample.word2token_start[str(entity[0])]
                entity[0] = sample.token2char_start[str(token_start)]
                token_end = sample.word2token_end[str(entity[1] - 1)]
                entity[1] = sample.token2char_end[str(token_end)]
                entities.append(entity)
            sample.predicted_entities = entities
            for triplet in sample.predicted_relations:
                triplet["subject"] = (
                    sample.token2char_start[
                        str(sample.word2token_start[str(triplet["subject"][0])])
                    ],
                    sample.token2char_end[
                        str(sample.word2token_end[str(triplet["subject"][1] - 1)])
                    ],
                    triplet["subject"][2],
                )
                triplet["object"] = (
                    sample.token2char_start[
                        str(sample.word2token_start[str(triplet["object"][0])])
                    ],
                    sample.token2char_end[
                        str(sample.word2token_end[str(triplet["object"][1] - 1)])
                    ],
                    triplet["object"][2],
                )
                # triplet["object"][0] = sample.token2char_start[
                #     str(sample.word2token_start[str(triplet["object"][0])])
                # ]
                # triplet["object"][1] = sample.token2char_end[
                #     str(sample.word2token_end[str(triplet["object"][1] - 1)])
                # ]

            sample = RelikREDataset._new_output_format(sample)

        return sample

    @staticmethod
    def merge_patches_predictions(sample) -> None:
        pass


def main():
    special_symbols = [NME_SYMBOL] + [f"R-{i}" for i in range(50)]

    relik_dataset = RelikREDataset(
        "/home/huguetcabot/alby-re/alby/data/nyt-alby+/valid.jsonl",
        materialize_samples=False,
        transformer_model="microsoft/deberta-v3-base",
        special_symbols=special_symbols,
        shuffle_candidates=False,
        flip_candidates=False,
        for_inference=True,
    )

    for batch in relik_dataset:
        print(batch)
        exit(0)


if __name__ == "__main__":
    main()
