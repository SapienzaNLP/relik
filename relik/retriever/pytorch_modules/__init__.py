from dataclasses import dataclass

import torch

from relik.retriever.indexers.document import Document

PRECISION_MAP = {
    None: torch.float32,
    32: torch.float32,
    16: torch.float16,
    torch.float32: torch.float32,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float": torch.float32,
    "half": torch.float16,
    "32": torch.float32,
    "16": torch.float16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


@dataclass
class RetrievedSample:
    """
    Dataclass for the output of the GoldenRetriever model.
    """

    score: float
    document: Document
