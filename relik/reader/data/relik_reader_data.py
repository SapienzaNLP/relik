import logging
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer

from relik.reader.data.relik_reader_data_utils import (
    add_noise_to_value,
    batchify,
    chunks,
    flatten,
)
from relik.reader.data.relik_reader_sample import (
    RelikReaderSample,
    load_relik_reader_samples,
)
from relik.reader.utils.special_symbols import NME_SYMBOL

logger = logging.getLogger(__name__)


def preprocess_sample(
    relik_sample: RelikReaderSample,
    tokenizer,
    lowercase_policy: float,
    add_topic: bool = False,
) -> None:
    if len(relik_sample.tokens) == 0:
        return

    if lowercase_policy > 0:
        lc_tokens = np.random.uniform(0, 1, len(relik_sample.tokens)) < lowercase_policy
        relik_sample.tokens = [
            t.lower() if lc else t for t, lc in zip(relik_sample.tokens, lc_tokens)
        ]

    tokenization_out = tokenizer(
        relik_sample.tokens,
        return_offsets_mapping=True,
        add_special_tokens=False,
    )

    window_tokens = tokenization_out.input_ids
    window_tokens = flatten(window_tokens)

    offsets_mapping = [
        [
            (
                ss + relik_sample.token2char_start[str(i)],
                se + relik_sample.token2char_start[str(i)],
            )
            for ss, se in tokenization_out.offset_mapping[i]
        ]
        for i in range(len(relik_sample.tokens))
    ]

    offsets_mapping = flatten(offsets_mapping)

    assert len(offsets_mapping) == len(window_tokens)

    window_tokens = [tokenizer.cls_token_id] + window_tokens + [tokenizer.sep_token_id]

    topic_offset = 0
    if add_topic:
        topic_tokens = tokenizer(
            relik_sample.doc_topic, add_special_tokens=False
        ).input_ids
        topic_offset = len(topic_tokens)
        relik_sample.topic_tokens = topic_offset
        window_tokens = window_tokens[:1] + topic_tokens + window_tokens[1:]

    token2char_start = {
        str(i): s for i, (s, _) in enumerate(offsets_mapping, start=topic_offset)
    }
    token2char_end = {
        str(i): e for i, (_, e) in enumerate(offsets_mapping, start=topic_offset)
    }
    token2word_start = {
        str(i): int(relik_sample._d["char2token_start"][str(s)])
        for i, (s, _) in enumerate(offsets_mapping, start=topic_offset)
        if str(s) in relik_sample._d["char2token_start"]
    }
    token2word_end = {
        str(i): int(relik_sample._d["char2token_end"][str(e)])
        for i, (_, e) in enumerate(offsets_mapping, start=topic_offset)
        if str(e) in relik_sample._d["char2token_end"]
    }
    relik_sample._d.update(
        dict(
            tokens=window_tokens,
            token2char_start=token2char_start,
            token2char_end=token2char_end,
            token2word_start=token2word_start,
            token2word_end=token2word_end,
        )
    )

    if "window_labels" in relik_sample._d:
        relik_sample.window_labels = [
            (s, e, l.replace("_", " ")) for s, e, l in relik_sample.window_labels
        ]


class TokenizationOutput(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    prediction_mask: torch.Tensor
    special_symbols_mask: torch.Tensor


class RelikDataset(IterableDataset):
    def __init__(
        self,
        dataset_path: Optional[str],
        materialize_samples: bool,
        transformer_model: Union[str, PreTrainedTokenizer],
        special_symbols: List[str],
        shuffle_candidates: Optional[Union[bool, float]] = False,
        for_inference: bool = False,
        noise_param: float = 0.1,
        sorting_fields: Optional[str] = None,
        tokens_per_batch: int = 2048,
        batch_size: int = None,
        max_batch_size: int = 128,
        section_size: int = 50_000,
        prebatch: bool = True,
        random_drop_gold_candidates: float = 0.0,
        use_nme: bool = True,
        max_subwords_per_candidate: bool = 22,
        mask_by_instances: bool = False,
        min_length: int = 5,
        max_length: int = 2048,
        model_max_length: int = 1000,
        split_on_cand_overload: bool = True,
        skip_empty_training_samples: bool = False,
        drop_last: bool = False,
        samples: Optional[Iterator[RelikReaderSample]] = None,
        lowercase_policy: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_path = dataset_path
        self.materialize_samples = materialize_samples
        self.samples: Optional[List[RelikReaderSample]] = None
        if self.materialize_samples:
            self.samples = list()

        if isinstance(transformer_model, str):
            self.tokenizer = self._build_tokenizer(transformer_model, special_symbols)
        else:
            self.tokenizer = transformer_model
        self.special_symbols = special_symbols
        self.shuffle_candidates = shuffle_candidates
        self.for_inference = for_inference
        self.noise_param = noise_param
        self.batching_fields = ["input_ids"]
        self.sorting_fields = (
            sorting_fields if sorting_fields is not None else self.batching_fields
        )

        self.tokens_per_batch = tokens_per_batch
        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.section_size = section_size
        self.prebatch = prebatch

        self.random_drop_gold_candidates = random_drop_gold_candidates
        self.use_nme = use_nme
        self.max_subwords_per_candidate = max_subwords_per_candidate
        self.mask_by_instances = mask_by_instances
        self.min_length = min_length
        self.max_length = max_length
        self.model_max_length = (
            model_max_length
            if model_max_length < self.tokenizer.model_max_length
            else self.tokenizer.model_max_length
        )

        # retrocompatibility workaround
        self.transformer_model = (
            transformer_model
            if isinstance(transformer_model, str)
            else transformer_model.name_or_path
        )
        self.split_on_cand_overload = split_on_cand_overload
        self.skip_empty_training_samples = skip_empty_training_samples
        self.drop_last = drop_last
        self.lowercase_policy = lowercase_policy
        self.samples = samples

    def _build_tokenizer(self, transformer_model: str, special_symbols: List[str]):
        return AutoTokenizer.from_pretrained(
            transformer_model,
            additional_special_tokens=[ss for ss in special_symbols],
            add_prefix_space=True,
        )

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
            "start_labels": lambda x: batchify(x, padding_value=-100),
            "end_labels": lambda x: batchify(x, padding_value=-100),
            "predictable_candidates_symbols": None,
            "predictable_candidates": None,
            "patch_offset": None,
            "optimus_labels": None,
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

    def _get_special_symbols_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        special_symbols_mask = input_ids >= (
            len(self.tokenizer) - len(self.special_symbols)
        )
        special_symbols_mask[0] = True
        return special_symbols_mask

    def _build_tokenizer_essentials(
        self, input_ids, original_sequence, sample
    ) -> TokenizationOutput:
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)

        total_sequence_len = len(input_ids)
        predictable_sentence_len = len(original_sequence)

        # token type ids
        token_type_ids = torch.cat(
            [
                input_ids.new_zeros(
                    predictable_sentence_len + 2
                ),  # original sentence bpes + CLS and SEP
                input_ids.new_ones(total_sequence_len - predictable_sentence_len - 2),
            ]
        )

        # prediction mask -> boolean on tokens that are predictable

        prediction_mask = torch.tensor(
            [1]
            + ([0] * predictable_sentence_len)
            + ([1] * (total_sequence_len - predictable_sentence_len - 1))
        )

        # add topic tokens to the prediction mask so that they cannot be predicted
        # or optimized during training
        topic_tokens = getattr(sample, "topic_tokens", None)
        if topic_tokens is not None:
            prediction_mask[1 : 1 + topic_tokens] = 1

        # If mask by instances is active the prediction mask is applied to everything
        # that is not indicated as an instance in the training set.
        if self.mask_by_instances:
            char_start2token = {
                cs: int(tok) for tok, cs in sample.token2char_start.items()
            }
            char_end2token = {ce: int(tok) for tok, ce in sample.token2char_end.items()}
            instances_mask = torch.ones_like(prediction_mask)
            for _, span_info in sample.instance_id2span_data.items():
                span_info = span_info[0]
                token_start = char_start2token[span_info[0]] + 1  # +1 for the CLS
                token_end = char_end2token[span_info[1]] + 1  # +1 for the CLS
                instances_mask[token_start : token_end + 1] = 0

            prediction_mask += instances_mask
            prediction_mask[prediction_mask > 1] = 1

        assert len(prediction_mask) == len(input_ids)

        # special symbols mask
        special_symbols_mask = self._get_special_symbols_mask(input_ids)

        return TokenizationOutput(
            input_ids,
            attention_mask,
            token_type_ids,
            prediction_mask,
            special_symbols_mask,
        )

    def _build_labels(
        self,
        sample,
        tokenization_output: TokenizationOutput,
        predictable_candidates: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        start_labels = [0] * len(tokenization_output.input_ids)
        end_labels = [0] * len(tokenization_output.input_ids)

        char_start2token = {v: int(k) for k, v in sample.token2char_start.items()}
        char_end2token = {v: int(k) for k, v in sample.token2char_end.items()}
        for cs, ce, gold_candidate_title in sample.window_labels:
            if gold_candidate_title not in predictable_candidates:
                if self.use_nme:
                    gold_candidate_title = NME_SYMBOL
                else:
                    continue
            # +1 is to account for the CLS token
            start_bpe = char_start2token[cs] + 1
            end_bpe = char_end2token[ce] + 1
            class_index = predictable_candidates.index(gold_candidate_title)
            if (
                start_labels[start_bpe] == 0 and end_labels[end_bpe] == 0
            ):  # prevent from having entities that ends with the same label
                start_labels[start_bpe] = class_index + 1  # +1 for the NONE class
                end_labels[end_bpe] = class_index + 1  # +1 for the NONE class
            else:
                print(
                    "Found entity with the same last subword, it will not be included."
                )
                print(
                    cs,
                    ce,
                    gold_candidate_title,
                    start_labels,
                    end_labels,
                    sample.doc_id,
                )

        ignored_labels_indices = tokenization_output.prediction_mask == 1

        start_labels = torch.tensor(start_labels, dtype=torch.long)
        start_labels[ignored_labels_indices] = -100

        end_labels = torch.tensor(end_labels, dtype=torch.long)
        end_labels[ignored_labels_indices] = -100

        return start_labels, end_labels

    def produce_sample_bag(
        self, sample, predictable_candidates: List[str], candidates_starting_offset: int
    ) -> Optional[Tuple[dict, list, int]]:
        # input sentence tokenization
        input_subwords = sample.tokens[1:-1]  # removing special tokens
        candidates_symbols = self.special_symbols[candidates_starting_offset:]

        predictable_candidates = list(predictable_candidates)
        original_predictable_candidates = list(predictable_candidates)

        # add NME as a possible candidate
        if self.use_nme:
            predictable_candidates.insert(0, NME_SYMBOL)

        # candidates encoding
        candidates_symbols = candidates_symbols[: len(predictable_candidates)]
        candidates_encoding_result = self.tokenizer.batch_encode_plus(
            [
                "{} {}".format(cs, ct) if ct != NME_SYMBOL else NME_SYMBOL
                for cs, ct in zip(candidates_symbols, predictable_candidates)
            ],
            add_special_tokens=False,
        ).input_ids

        if (
            self.max_subwords_per_candidate is not None
            and self.max_subwords_per_candidate > 0
        ):
            candidates_encoding_result = [
                cer[: self.max_subwords_per_candidate]
                for cer in candidates_encoding_result
            ]

        # drop candidates if the number of input tokens is too long for the model
        if (
            sum(map(len, candidates_encoding_result))
            + len(input_subwords)
            + 20  # + 20 special tokens
            > self.model_max_length
        ):
            acceptable_tokens_from_candidates = (
                self.model_max_length - 20 - len(input_subwords)
            )
            i = 0
            cum_len = 0
            while (
                cum_len + len(candidates_encoding_result[i])
                < acceptable_tokens_from_candidates
            ):
                cum_len += len(candidates_encoding_result[i])
                i += 1

            candidates_encoding_result = candidates_encoding_result[:i]
            candidates_symbols = candidates_symbols[:i]
            predictable_candidates = predictable_candidates[:i]

        # final input_ids build
        input_ids = self._build_input_ids(
            sentence_input_ids=input_subwords,
            candidates_input_ids=candidates_encoding_result,
        )

        # complete input building (e.g. attention / prediction mask)
        tokenization_output = self._build_tokenizer_essentials(
            input_ids, input_subwords, sample
        )

        output_dict = {
            "input_ids": tokenization_output.input_ids,
            "attention_mask": tokenization_output.attention_mask,
            "token_type_ids": tokenization_output.token_type_ids,
            "prediction_mask": tokenization_output.prediction_mask,
            "special_symbols_mask": tokenization_output.special_symbols_mask,
            "sample": sample,
            "predictable_candidates_symbols": candidates_symbols,
            "predictable_candidates": predictable_candidates,
        }

        # labels creation
        if sample.window_labels is not None:
            start_labels, end_labels = self._build_labels(
                sample,
                tokenization_output,
                predictable_candidates,
            )
            output_dict.update(start_labels=start_labels, end_labels=end_labels)

        if (
            "roberta" in self.transformer_model
            or "longformer" in self.transformer_model
        ):
            del output_dict["token_type_ids"]

        predictable_candidates_set = set(predictable_candidates)
        remaining_candidates = [
            candidate
            for candidate in original_predictable_candidates
            if candidate not in predictable_candidates_set
        ]
        total_used_candidates = (
            candidates_starting_offset
            + len(predictable_candidates)
            - (1 if self.use_nme else 0)
        )

        if self.use_nme:
            assert predictable_candidates[0] == NME_SYMBOL

        return output_dict, remaining_candidates, total_used_candidates

    def __iter__(self):
        dataset_iterator = self.dataset_iterator_func()

        current_dataset_elements = []

        i = None
        for i, dataset_elem in enumerate(dataset_iterator, start=1):
            if (
                self.section_size is not None
                and len(current_dataset_elements) == self.section_size
            ):
                for batch in self.materialize_batches(current_dataset_elements):
                    yield batch
                current_dataset_elements = []

            current_dataset_elements.append(dataset_elem)

            if i % 50_000 == 0:
                logger.info(f"Processed: {i} number of elements")

        if len(current_dataset_elements) != 0:
            for batch in self.materialize_batches(current_dataset_elements):
                yield batch

        if i is None:
            # logger.debug(f"Dataset finished: {i} number of elements processed")
        # else:
            logger.warning("Dataset empty")

    def dataset_iterator_func(self):
        skipped_instances = 0
        data_samples = (
            load_relik_reader_samples(self.dataset_path)
            if self.samples is None
            else self.samples
        )
        for sample in data_samples:
            preprocess_sample(
                sample, self.tokenizer, lowercase_policy=self.lowercase_policy
            )
            current_patch = 0
            sample_bag, used_candidates = None, None
            # TODO: compatibility shit
            # sample.window_candidates = sample.span_candidates
            remaining_candidates = list(sample.span_candidates)

            if not self.for_inference:
                # randomly drop gold candidates at training time
                if (
                    self.random_drop_gold_candidates > 0.0
                    and np.random.uniform() < self.random_drop_gold_candidates
                    and len(set(ct for _, _, ct in sample.window_labels)) > 1
                ):
                    # selecting candidates to drop
                    np.random.shuffle(sample.window_labels)
                    n_dropped_candidates = np.random.randint(
                        0, len(sample.window_labels) - 1
                    )
                    dropped_candidates = [
                        label_elem[-1]
                        for label_elem in sample.window_labels[:n_dropped_candidates]
                    ]
                    dropped_candidates = set(dropped_candidates)

                    # saving NMEs because they should not be dropped
                    if NME_SYMBOL in dropped_candidates:
                        dropped_candidates.remove(NME_SYMBOL)

                    # sample update
                    sample.window_labels = [
                        (
                            (s, e, _l)
                            if _l not in dropped_candidates
                            else (s, e, NME_SYMBOL)
                        )
                        for s, e, _l in sample.window_labels
                    ]
                    remaining_candidates = [
                        wc
                        for wc in remaining_candidates
                        if wc not in dropped_candidates
                    ]

                # shuffle candidates
                if (
                    isinstance(self.shuffle_candidates, bool)
                    and self.shuffle_candidates
                ) or (
                    isinstance(self.shuffle_candidates, float)
                    and np.random.uniform() < self.shuffle_candidates
                ):
                    np.random.shuffle(remaining_candidates)

            while len(remaining_candidates) != 0:
                sample_bag = self.produce_sample_bag(
                    sample,
                    predictable_candidates=remaining_candidates,
                    candidates_starting_offset=(
                        used_candidates if used_candidates is not None else 0
                    ),
                )
                if sample_bag is not None:
                    sample_bag, remaining_candidates, used_candidates = sample_bag
                    if (
                        self.for_inference
                        or not self.skip_empty_training_samples
                        or (
                            (
                                sample_bag.get("start_labels") is not None
                                and torch.any(sample_bag["start_labels"] > 1).item()
                            )
                            or (
                                sample_bag.get("optimus_labels") is not None
                                and len(sample_bag["optimus_labels"]) > 0
                            )
                        )
                    ):
                        sample_bag["patch_offset"] = current_patch
                        current_patch += 1
                        yield sample_bag
                    else:
                        skipped_instances += 1
                        if skipped_instances % 1000 == 0 and skipped_instances != 0:
                            logger.info(
                                f"Skipped {skipped_instances} instances since they did not have any gold labels..."
                            )

                # Just use the first fitting candidates if split on
                #  cand is not True
                if not self.split_on_cand_overload:
                    break

    def preshuffle_elements(self, dataset_elements: List):
        # This shuffling is done so that when using the sorting function,
        # if it is deterministic given a collection and its order, we will
        # make the whole operation not deterministic anymore.
        # Basically, the aim is not to build every time the same batches.
        if not self.for_inference:
            dataset_elements = np.random.permutation(dataset_elements)

        def sorting_fn(elem):
            return (
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

        ds = list(chunks(dataset_elements, 64))
        np.random.shuffle(ds)
        return flatten(ds)

    def materialize_batches(
        self, dataset_elements: List[Dict[str, Any]]
    ) -> Generator[Dict[str, Any], None, None]:
        if self.prebatch:
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
    def convert_to_char_annotations(
        sample: RelikReaderSample,
        remove_nmes: bool = True,
    ) -> RelikReaderSample:
        """
        Converts the annotations to char annotations.

        Args:
            sample (:obj:`RelikReaderSample`):
                The sample to convert.
            remove_nmes (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to remove the NMEs from the annotations.
        Returns:
            :obj:`RelikReaderSample`: The converted sample.
        """
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
            # TODO: which one is kept if there are multiple candidates with same title?
            # and where is the order?
            char_probs_annotations[(span_start, span_end)] = candidates_probs

        sample.predicted_window_labels_chars = char_annotations
        sample.probs_window_labels_chars = char_probs_annotations

        # try-out for a new format
        sample.predicted_spans = char_annotations
        sample.predicted_spans_probabilities = char_probs_annotations

        return sample

    @staticmethod
    def convert_to_word_annotations(
        sample: RelikReaderSample,
        remove_nmes: bool = True,
    ) -> RelikReaderSample:
        """
        Converts the annotations to tokens annotations.

        Args:
            sample (:obj:`RelikReaderSample`):
                The sample to convert.
            remove_nmes (:obj:`bool`, `optional`, defaults to :obj:`True`):
                Whether to remove the NMEs from the annotations.

        Returns:
            :obj:`RelikReaderSample`: The converted sample.
        """
        word_annotations = set()
        for (
            predicted_entity,
            predicted_spans,
        ) in sample.predicted_window_labels.items():
            if predicted_entity == NME_SYMBOL and remove_nmes:
                continue

            for span_start, span_end in predicted_spans:
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

                span_start = sample.token2word_start[str(span_start)]
                span_end = sample.token2word_end[str(span_end)]

                word_annotations.add((span_start, span_end + 1, predicted_entity))

        word_probs_annotations = dict()
        for (
            span_start,
            span_end,
        ), candidates_probs in sample.span_title_probabilities.items():
            for span_start, span_end in predicted_spans:
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
            span_start = sample.token2word_start[str(span_start)]
            span_end = sample.token2word_end[str(span_end)]
            word_probs_annotations[(span_start, span_end + 1)] = {
                title for title, _ in candidates_probs
            }

        sample.predicted_window_labels_words = word_annotations
        sample.probs_window_labels_words = word_probs_annotations

        # try-out for a new format
        sample.predicted_spans = word_annotations
        sample.predicted_spans_probabilities = word_probs_annotations
        return sample

    @staticmethod
    def merge_patches_predictions(sample) -> None:
        sample._d["predicted_window_labels"] = dict()
        predicted_window_labels = sample._d["predicted_window_labels"]

        sample._d["span_title_probabilities"] = dict()
        span_title_probabilities = sample._d["span_title_probabilities"]

        span2title = dict()
        for _, patch_info in sorted(sample.patches.items(), key=lambda x: x[0]):
            # selecting span predictions
            for predicted_title, predicted_spans in patch_info[
                "predicted_window_labels"
            ].items():
                for pred_span in predicted_spans:
                    pred_span = tuple(pred_span)
                    curr_title = span2title.get(pred_span)
                    if curr_title is None or curr_title == NME_SYMBOL:
                        span2title[pred_span] = predicted_title
                    # else:
                    #     print("Merging at patch level")

            # selecting span predictions probability
            for predicted_span, titles_probabilities in patch_info[
                "span_title_probabilities"
            ].items():
                if predicted_span not in span_title_probabilities:
                    span_title_probabilities[predicted_span] = titles_probabilities

        for span, title in span2title.items():
            if title not in predicted_window_labels:
                predicted_window_labels[title] = list()
            predicted_window_labels[title].append(span)
