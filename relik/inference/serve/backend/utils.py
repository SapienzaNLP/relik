import ast
import os
from dataclasses import dataclass


@dataclass
class ServerParameterManager:
    relik_pretrained: str = os.environ.get("RELIK_PRETRAINED", None)
    device: str = os.environ.get("DEVICE", "cpu")
    retriever_device: str | None = os.environ.get("RETRIEVER_DEVICE", None)
    document_index_device: str | None = os.environ.get("INDEX_DEVICE", None)
    reader_device: str | None = os.environ.get("READER_DEVICE", None)
    precision: int | str | None = os.environ.get("PRECISION", "fp32")
    retriever_precision: int | str | None = os.environ.get("RETRIEVER_PRECISION", None)
    document_index_precision: int | str | None = os.environ.get("INDEX_PRECISION", None)
    reader_precision: int | str | None = os.environ.get("READER_PRECISION", None)
    annotation_type: str = os.environ.get("ANNOTATION_TYPE", "char")
    question_encoder: str = os.environ.get("QUESTION_ENCODER", None)
    passage_encoder: str = os.environ.get("PASSAGE_ENCODER", None)
    document_index: str = os.environ.get("DOCUMENT_INDEX", None)
    reader_encoder: str = os.environ.get("READER_ENCODER", None)
    top_k: int = int(os.environ.get("TOP_K", 100))
    use_faiss: bool = os.environ.get("USE_FAISS", False)
    retriever_batch_size: int = int(os.environ.get("RETRIEVER_BATCH_SIZE", 32))
    reader_batch_size: int = int(os.environ.get("READER_BATCH_SIZE", 32))
    window_size: int = int(os.environ.get("WINDOW_SIZE", 32))
    window_stride: int = int(os.environ.get("WINDOW_SIZE", 16))
    split_on_spaces: bool = os.environ.get("SPLIT_ON_SPACES", False)
    # relik_config_override: dict = ast.literal_eval(
    #     os.environ.get("RELIK_CONFIG_OVERRIDE", None)
    # )


class RayParameterManager:
    def __init__(self) -> None:
        self.num_gpus = int(os.environ.get("NUM_GPUS", 1))
        self.min_replicas = int(os.environ.get("MIN_REPLICAS", 1))
        self.max_replicas = int(os.environ.get("MAX_REPLICAS", 1))
