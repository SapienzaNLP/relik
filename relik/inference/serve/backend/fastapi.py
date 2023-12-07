import logging
import os
from pathlib import Path
from typing import List, Union
import psutil

import torch

from relik.common.utils import is_package_available
from relik.inference.annotator import Relik

if not is_package_available("fastapi"):
    raise ImportError(
        "FastAPI is not installed. Please install FastAPI with `pip install relik[serve]`."
    )
from fastapi import FastAPI, HTTPException, APIRouter


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
        )

        self.router = APIRouter()
        self.router.add_api_route("/api/relik", self.relik_endpoint, methods=["POST"])

        logger.info("RelikServer initialized.")

    # @serve.batch()
    async def __call__(self, text: List[str]) -> List:
        return self.relik(text, annotation_type=self.annotation_type)

    # @app.post("/api/relik")
    async def relik_endpoint(self, text: Union[str, List[str]]):
        try:
            # get predictions for the retriever
            return await self(text)
        except Exception as e:
            # log the entire stack trace
            logger.exception(e)
            raise HTTPException(status_code=500, detail=f"Server Error: {e}")


app = FastAPI(
    title="ReLiK",
    version=VERSION["VERSION"],
    description="ReLiK REST API",
)
server = RelikServer(**vars(SERVER_MANAGER))
app.include_router(server.router)
