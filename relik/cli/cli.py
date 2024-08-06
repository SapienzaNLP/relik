import ast
import json
import os
import subprocess
from pathlib import Path
from typing import Annotated, Optional, Union

import typer
from tqdm import tqdm

from relik.cli import data, reader, retriever
from relik.common.log import get_logger
from relik.common.utils import batch_generator
from relik.inference.annotator import Relik
from relik.inference.data.splitters.blank_sentence_splitter import BlankSentenceSplitter
from relik.inference.data.splitters.spacy_sentence_splitter import SpacySentenceSplitter
from relik.inference.data.splitters.window_based_splitter import WindowSentenceSplitter
from relik.inference.data.tokenizers.spacy_tokenizer import SpacyTokenizer
from relik.inference.data.window.manager import WindowManager
from relik.inference.serve.backend.fastapi_be import main as serve_fastapi
from relik.inference.serve.frontend.gradio_fe import main as serve_gradio

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
app.add_typer(data.app, name="data")
app.add_typer(reader.app, name="reader")
app.add_typer(retriever.app, name="retriever")


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def inference(
    model_name_or_path: str,
    input_path: str,
    output_path: str,
    batch_size: int = 8,
    num_workers: int = 4,
    device: str = "cuda",
    precision: str = "fp16",
    top_k: int = 100,
    io_batch_size: int = 1_000,
    annotation_type: str = "char",
    progress_bar: bool = True,
    model_kwargs: str = None,
    inference_kwargs: str = None,
):
    """
    Perform inference on raw text files using a pre-trained pipeline.

    Args:
        model_name_or_path (str):
            The model name or path to the model directory.
        input_path (str):
            The path to the input file as txt. Each line in the file should contain a single text sample.
        output_path (str):
            The path to the output file as jsonl. Each line in the file will contain the model's prediction.
        batch_size (int, optional):
            The batch size for inference. Defaults to 8.
        num_workers (int, optional):
            The number of workers for parallel processing. Defaults to 4.
        device (str, optional):
            The device to use for inference (e.g., "cuda", "cpu"). Defaults to "cuda".
        precision (str, optional):
            The precision mode for inference (e.g., "fp16", "fp32"). Defaults to "fp16".
        top_k (int, optional):
            The number of top predictions of the retriever to consider. Defaults to 100.
        window_size (int, optional):
            The window size for sliding window annotation. Defaults to None.
        window_stride (int, optional):
            The stride size for sliding window annotation. Defaults to None.
        annotation_type (str, optional):
            The type of annotation to use (e.g., "CHAR", "WORD"). Defaults to "CHAR".
        progress_bar (bool, optional):
            Whether to display a progress bar during inference. Defaults to True.
        model_kwargs (dict, optional):
            Additional keyword arguments for the model. Defaults to None.
        inference_kwargs (dict, optional):
            Additional keyword arguments for inference. Defaults to None.

    """
    if model_kwargs is None:
        model_kwargs = {}
    else:
        model_kwargs = ast.literal_eval(model_kwargs)

    model_kwargs.update(dict(device=device, precision=precision))

    if inference_kwargs is None:
        inference_kwargs = {}
    else:
        inference_kwargs = ast.literal_eval(inference_kwargs)

    inference_kwargs.update(
        dict(
            num_workers=num_workers,
            top_k=top_k,
            annotation_type=annotation_type,
            progress_bar=progress_bar,
        )
    )
    if "retriever_batch_size" not in inference_kwargs:
        inference_kwargs["retriever_batch_size"] = batch_size
    if "reader_batch_size" not in inference_kwargs:
        inference_kwargs["reader_batch_size"] = batch_size

    # create folder if it doesn't exist
    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_dir():
        input_files = list(input_path.glob("*.txt"))
    else:
        input_files = [input_path]
    if output_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    relik = Relik.from_pretrained(model_name_or_path, **model_kwargs)

    for i, path in enumerate(input_files):
        logger.info("Starting annotation for %s", path)
        if output_path.is_dir():
            output_path = output_path / f"{path.stem}.jsonl"
            out_context = open(output_path, "w")
        else:
            # if we are writing to a single file, we need to append
            # but only if we are doing inference on multiple files
            # and if we are at the second file
            out_context = open(output_path, "w" if i == 0 else "a")
        with open(path, "r") as f_in, out_context as f_out:
            for batch in tqdm(batch_generator(f_in, io_batch_size)):
                # clean batch
                batch = [line.strip() for line in batch if line.strip()]
                predictions = relik(batch, **inference_kwargs)
                for prediction in predictions:
                    prdiction_dict = prediction.to_dict()
                    prediction_dict = {
                        "text": prdiction_dict["text"],
                        "spans": prdiction_dict["spans"],
                    }
                    f_out.write(json.dumps(prediction_dict) + "\n")


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def serve(
    relik_pretrained: Annotated[
        str,
        typer.Argument(help="The device to use for relik (e.g., 'cuda', 'cpu')."),
    ],
    device: str = "cpu",
    retriever_device: str = None,
    document_index_device: str = None,
    reader_device: str = None,
    precision: int = 32,
    retriever_precision: int = None,
    document_index_precision: int = None,
    reader_precision: int = None,
    annotation_type: str = "char",
    host: str = "0.0.0.0",
    port: int = 8000,
    frontend: bool = False,
):
    serve_fastapi(
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
        host=host,
        port=port,
        frontend=frontend,
    )


if __name__ == "__main__":
    app()
