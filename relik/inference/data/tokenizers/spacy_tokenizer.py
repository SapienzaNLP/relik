import logging
from copy import deepcopy
from typing import Dict, List, Tuple, Union, Any

import spacy

# from ipa.common.utils import load_spacy
from spacy.cli.download import download as spacy_download
from spacy.tokens import Doc

from relik.common.log import get_logger
from relik.inference.data.objects import Word
from relik.inference.data.tokenizers import SPACY_LANGUAGE_MAPPER
from relik.inference.data.tokenizers.base_tokenizer import BaseTokenizer

logger = get_logger(level=logging.DEBUG)

# Spacy and Stanza stuff

LOADED_SPACY_MODELS: Dict[Tuple[str, bool, bool, bool, bool], spacy.Language] = {}


def load_spacy(
    language: str,
    pos_tags: bool = False,
    lemma: bool = False,
    parse: bool = False,
    split_on_spaces: bool = False,
) -> spacy.Language:
    """
    Download and load spacy model.

    Args:
        language (:obj:`str`, defaults to :obj:`en`):
            Language of the text to tokenize.
        pos_tags (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs POS tagging with spacy model.
        lemma (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs lemmatization with spacy model.
        parse (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs dependency parsing with spacy model.
        split_on_spaces (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, will split by spaces without performing tokenization.

    Returns:
        :obj:`spacy.Language`: The spacy model loaded.
    """
    exclude = ["vectors", "textcat", "ner"]
    if not pos_tags:
        exclude.append("tagger")
    if not lemma:
        exclude.append("lemmatizer")
    if not parse:
        exclude.append("parser")

    # check if the model is already loaded
    # if so, there is no need to reload it
    spacy_params = (language, pos_tags, lemma, parse, split_on_spaces)
    if spacy_params not in LOADED_SPACY_MODELS:
        try:
            spacy_tagger = spacy.load(language, exclude=exclude)
        except OSError:
            logger.warning(
                "Spacy model '%s' not found. Downloading and installing.", language
            )
            spacy_download(language)
            spacy_tagger = spacy.load(language, exclude=exclude)

        # if everything is disabled, return only the tokenizer
        # for faster tokenization
        # TODO: is it really faster?
        # TODO: check split_on_spaces behaviour if we don't do this if
        if len(exclude) >= 6 and split_on_spaces:
            spacy_tagger = spacy_tagger.tokenizer
        LOADED_SPACY_MODELS[spacy_params] = spacy_tagger

    return LOADED_SPACY_MODELS[spacy_params]


class SpacyTokenizer(BaseTokenizer):
    """
    A :obj:`Tokenizer` that uses SpaCy to tokenizer and preprocess the text. It returns :obj:`Word` objects.

    Args:
        language (:obj:`str`, optional, defaults to :obj:`en`):
            Language of the text to tokenize.
        return_pos_tags (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs POS tagging with spacy model.
        return_lemmas (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs lemmatization with spacy model.
        return_deps (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, performs dependency parsing with spacy model.
        use_gpu (:obj:`bool`, optional, defaults to :obj:`False`):
            If :obj:`True`, will load the Stanza model on GPU.
    """

    def __init__(
        self,
        language: str = "en",
        return_pos_tags: bool = False,
        return_lemmas: bool = False,
        return_deps: bool = False,
        use_gpu: bool = False,
    ):
        super().__init__()
        if language not in SPACY_LANGUAGE_MAPPER:
            raise ValueError(
                f"`{language}` language not supported. The supported "
                f"languages are: {list(SPACY_LANGUAGE_MAPPER.keys())}."
            )
        if use_gpu:
            # load the model on GPU
            # if the GPU is not available or not correctly configured,
            # it will rise an error
            spacy.require_gpu()
        self.spacy = load_spacy(
            SPACY_LANGUAGE_MAPPER[language],
            return_pos_tags,
            return_lemmas,
            return_deps,
        )

    def __call__(
        self,
        texts: Union[str, List[str], List[List[str]]],
        is_split_into_words: bool = False,
        **kwargs,
    ) -> Union[List[Word], List[List[Word]]]:
        """
        Tokenize the input into single words using SpaCy models.

        Args:
            texts (:obj:`str`, :obj:`List[str]`, :obj:`List[List[str]]`):
                Text to tag. It can be a single string, a batch of string and pre-tokenized strings.
            is_split_into_words (:obj:`bool`, optional, defaults to :obj:`False`):
                If :obj:`True` and the input is a string, the input is split on spaces.

        Returns:
            :obj:`List[List[Word]]`: The input text tokenized in single words.

        Example::

            >>> from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer

            >>> spacy_tokenizer = SpacyTokenizer(language="en", pos_tags=True, lemma=True)
            >>> spacy_tokenizer("Mary sold the car to John.")

        """
        # check if input is batched or a single sample
        is_batched = self.check_is_batched(texts, is_split_into_words)

        if is_batched:
            tokenized = self.tokenize_batch(texts, is_split_into_words)
        else:
            tokenized = self.tokenize(texts, is_split_into_words)

        return tokenized

    def tokenize(self, text: Union[str, List[str]], is_split_into_words: bool) -> Doc:
        if is_split_into_words:
            if isinstance(text, str):
                text = text.split(" ")
            elif isinstance(text, list):
                text = text
            else:
                raise ValueError(
                    f"text must be either `str` or `list`, found: `{type(text)}`"
                )
            spaces = [True] * len(text)
            return self.spacy(Doc(self.spacy.vocab, words=text, spaces=spaces))
        return self.spacy(text)

    def tokenize_batch(
        self, texts: Union[List[str], List[List[str]]], is_split_into_words: bool
    ) -> list[Any] | list[Doc]:
        try:
            if is_split_into_words:
                if isinstance(texts[0], str):
                    texts = [text.split(" ") for text in texts]
                elif isinstance(texts[0], list):
                    texts = texts
                else:
                    raise ValueError(
                        f"text must be either `str` or `list`, found: `{type(texts[0])}`"
                    )
                spaces = [[True] * len(text) for text in texts]
                texts = [
                    Doc(self.spacy.vocab, words=text, spaces=space)
                    for text, space in zip(texts, spaces)
                ]
            return list(self.spacy.pipe(texts))
        except AttributeError:
            # a WhitespaceSpacyTokenizer has no `pipe()` method, we use simple for loop
            return [self.spacy(tokens) for tokens in texts]

