from argparse import ArgumentParser
import logging
import os
from pathlib import Path
from typing import Annotated, Dict, List, Union
import psutil

import torch
import uvicorn

from relik.common.utils import is_package_available
from relik.inference.annotator import Relik
from relik.inference.data.objects import AnnotationType, RelikOutput, TaskType
from relik.retriever.indexers.document import Document

if not is_package_available("fastapi"):
    raise ImportError(
        "FastAPI is not installed. Please install FastAPI with `pip install relik[serve]`."
    )
from fastapi import FastAPI, HTTPException, APIRouter, Query


from relik.common.log import get_logger
from relik.inference.serve.backend.utils import (
    RayParameterManager,
    ServerParameterManager,
)

logger = get_logger(__name__, level=logging.INFO)

VERSION = {}  # type: ignore
with open(
    Path(__file__).parent.parent.parent.parent / "version.py", "r"
) as version_file:
    exec(version_file.read(), VERSION)

# Env variables for server
SERVER_MANAGER = ServerParameterManager()
RAY_MANAGER = RayParameterManager()


class RelikServer:
    def __init__(
        self,
        relik_pretrained: str | None = None,
        device: str = "cpu",
        retriever_device: str | None = None,
        document_index_device: str | None = None,
        reader_device: str | None = None,
        precision: str | int | torch.dtype = 32,
        retriever_precision: str | int | torch.dtype | None = None,
        document_index_precision: str | int | torch.dtype | None = None,
        reader_precision: str | int | torch.dtype | None = None,
        annotation_type: str = "char",
        skip_metadata: bool = False,
        **kwargs,
    ):
        num_threads = os.getenv("TORCH_NUM_THREADS", psutil.cpu_count(logical=False))
        torch.set_num_threads(num_threads)
        logger.info(f"Torch is running on {num_threads} threads.")
        # parameters
        logger.info(f"RELIK_PRETRAINED: {relik_pretrained}")
        self.relik_pretrained = relik_pretrained
        logger.info(f"DEVICE: {device}")
        self.device = device
        if retriever_device is not None:
            logger.info(f"RETRIEVER_DEVICE: {retriever_device}")
        self.retriever_device = retriever_device or device
        if document_index_device is not None:
            logger.info(f"INDEX_DEVICE: {document_index_device}")
        self.document_index_device = document_index_device or retriever_device
        if reader_device is not None:
            logger.info(f"READER_DEVICE: {reader_device}")
        self.reader_device = reader_device
        logger.info(f"PRECISION: {precision}")
        self.precision = precision
        if retriever_precision is not None:
            logger.info(f"RETRIEVER_PRECISION: {retriever_precision}")
        self.retriever_precision = retriever_precision or precision
        if document_index_precision is not None:
            logger.info(f"INDEX_PRECISION: {document_index_precision}")
        self.document_index_precision = document_index_precision or precision
        if reader_precision is not None:
            logger.info(f"READER_PRECISION: {reader_precision}")
        self.reader_precision = reader_precision or precision
        logger.info(f"ANNOTATION_TYPE: {annotation_type}")
        self.annotation_type = annotation_type

        self.relik = Relik.from_pretrained(
            self.relik_pretrained,
            device=self.device,
            retriever_device=self.retriever_device,
            document_index_device=self.document_index_device,
            reader_device=self.reader_device,
            precision=self.precision,
            retriever_precision=self.retriever_precision,
            document_index_precision=self.document_index_precision,
            reader_precision=self.reader_precision,
            skip_metadata=skip_metadata,
        )

        self.router = APIRouter()
        self.router.add_api_route("/api/relik", self.relik_endpoint, methods=["GET"])

        logger.info("RelikServer initialized.")

    # @serve.batch()
    async def __call__(
        self,
        text: List[str],
        top_k: int | None = None,
        window_size: int | str | None = None,
        window_stride: int | str | None = None,
        is_split_into_words: bool = False,
        retriever_batch_size: int | None = 32,
        reader_batch_size: int | None = 32,
        return_windows: bool = False,
        use_doc_topic: bool = False,
        annotation_type: str | AnnotationType = AnnotationType.CHAR,
        relation_threshold: float = 0.5,
    ) -> List:
        output = self.relik(
            text=text,
            top_k=top_k,
            window_size=window_size,
            window_stride=window_stride,
            is_split_into_words=is_split_into_words,
            retriever_batch_size=retriever_batch_size,
            reader_batch_size=reader_batch_size,
            return_windows=return_windows,
            use_doc_topic=use_doc_topic,
            annotation_type=annotation_type,
            relation_threshold=relation_threshold,
        )
        output = output if isinstance(output, list) else [output]
        return [o.to_dict() for o in output]

    async def relik_endpoint(
        self,
        text: Annotated[list[str] | None, Query()] = None,
        top_k: int | None = None,
        window_size: int | str | None = None,
        window_stride: int | str | None = None,
        is_split_into_words: bool = False,
        retriever_batch_size: int | None = 32,
        reader_batch_size: int | None = 32,
        return_windows: bool = False,
        use_doc_topic: bool = False,
        annotation_type: str | AnnotationType = AnnotationType.CHAR,
        relation_threshold: float = 0.5,
    ) -> List:
        try:
            if window_size:
                # check if window size is a number as string
                if window_size.isdigit():
                    window_size = int(window_size)

            if window_stride:
                # check if window stride is a number as string
                if window_stride.isdigit():
                    window_stride = int(window_stride)

            # get predictions for the retriever
            return await self(
                text=text,
                top_k=top_k,
                window_size=window_size,
                window_stride=window_stride,
                is_split_into_words=is_split_into_words,
                retriever_batch_size=retriever_batch_size,
                reader_batch_size=reader_batch_size,
                return_windows=return_windows,
                use_doc_topic=use_doc_topic,
                annotation_type=annotation_type,
                relation_threshold=relation_threshold,
            )
        except Exception as e:
            # log the entire stack trace
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Server Error: {e}")


def main(
    relik_pretrained: str,
    device: str = "cpu",
    retriever_device: str = None,
    document_index_device: str = None,
    reader_device: str = None,
    precision: str = "32",
    retriever_precision: str = None,
    document_index_precision: str = None,
    reader_precision: str = None,
    annotation_type: str = "char",
    workers: int = None,
    host: str = "localhost",
    port: int = 8000,
    frontend: bool = False,
):
    app = FastAPI(
        title="ReLiK - A blazing fast and lightweight Information Extraction model for Entity Linking and Relation Extraction.",
        version=VERSION["VERSION"],
        description="ReLiK REST API",
    )
    server = RelikServer(
        relik_pretrained=relik_pretrained,
        device=device,
        retriever_device=retriever_device,
        document_index_device=document_index_device,
        reader_device=reader_device,
        precision=precision,
        retriever_precision=retriever_precision,
        document_index_precision=document_index_precision,
        reader_precision=reader_precision,
        annotation_type=annotation_type,
    )
    app.include_router(server.router)
    if frontend:
        from relik.inference.serve.frontend.gradio_fe import main as serve_frontend
        import threading

        threading.Thread(target=serve_frontend, daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level="info", workers=workers)


if __name__ == "__main__":

    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--relik_pretrained",
        type=str,
        help="Path to the pretrained ReLiK model",
    )
    arg_parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on"
    )
    arg_parser.add_argument(
        "--retriever_device",
        type=str,
        default=None,
        help="Device to run the retriever on",
    )
    arg_parser.add_argument(
        "--document_index_device",
        type=str,
        default=None,
        help="Device to run the document index on",
    )
    arg_parser.add_argument(
        "--reader_device", type=str, default=None, help="Device to run the reader on"
    )
    arg_parser.add_argument(
        "--precision", type=str, default=32, help="Precision of the model"
    )
    arg_parser.add_argument(
        "--retriever_precision",
        type=str,
        default=None,
        help="Precision of the retriever",
    )
    arg_parser.add_argument(
        "--document_index_precision",
        type=str,
        default=None,
        help="Precision of the document index",
    )
    arg_parser.add_argument(
        "--reader_precision", type=str, default=None, help="Precision of the reader"
    )
    arg_parser.add_argument(
        "--annotation_type", type=str, default="char", help="Type of annotation"
    )
    arg_parser.add_argument(
        "--workers", type=int, default=None, help="Number of workers"
    )
    arg_parser.add_argument(
        "--host", type=str, default="localhost", help="Host address"
    )
    arg_parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = arg_parser.parse_args()

    main(**vars(args))
