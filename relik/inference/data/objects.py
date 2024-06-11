from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NamedTuple, Optional

from relik.reader.pytorch_modules.hf.modeling_relik import RelikReaderSample
from relik.retriever.indexers.document import Document


@dataclass
class Word:
    """
    A word representation that includes text, index in the sentence, POS tag, lemma,
    dependency relation, and similar information.

    # Parameters
    text : `str`, optional
        The text representation.
    index : `int`, optional
        The word offset in the sentence.
    lemma : `str`, optional
        The lemma of this word.
    pos : `str`, optional
        The coarse-grained part of speech of this word.
    dep : `str`, optional
        The dependency relation for this word.

    input_id : `int`, optional
        Integer representation of the word, used to pass it to a model.
    token_type_id : `int`, optional
        Token type id used by some transformers.
    attention_mask: `int`, optional
        Attention mask used by transformers, indicates to the model which tokens should
        be attended to, and which should not.
    """

    text: str
    i: int
    idx: Optional[int] = None
    idx_end: Optional[int] = None
    # preprocessing fields
    lemma: Optional[str] = None
    pos: Optional[str] = None
    dep: Optional[str] = None
    head: Optional[int] = None

    def __str__(self):
        return self.text

    def __repr__(self):
        return self.__str__()


class Span(NamedTuple):
    start: int
    end: int
    label: str
    text: str


class Triples(NamedTuple):
    subject: Span
    label: str
    object: Span
    confidence: float


class Candidates(NamedTuple):
    span: Dict[List[Document]]
    triplet: Dict[List[Document]]


@dataclass
class RelikOutput:
    text: str
    tokens: List[str]
    spans: List[Span]
    triples: List[Triples]
    candidates: Candidates = None
    windows: Optional[List[RelikReaderSample]] = None

    # convert to dict
    def to_dict(self):
        self_dict = {
            "text": self.text,
            "tokens": self.tokens,
            "spans": self.spans,
            "triples": self.triples,
            "candidates": {
                "span": [
                    [[doc.to_dict() for doc in documents] for documents in window]
                    for window in self.candidates.span
                ],
                "triplet": [
                    [[doc.to_dict() for doc in documents] for documents in window]
                    for window in self.candidates.triplet
                ],
            },
        }
        if self.windows is not None:
            self_dict["windows"] = [window.to_dict() for window in self.windows]
        return self_dict


class AnnotationType(Enum):
    CHAR = "char"
    WORD = "word"


class TaskType(Enum):
    SPAN = "span"
    TRIPLET = "triplet"
    BOTH = "both"
