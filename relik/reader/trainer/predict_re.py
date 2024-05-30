import argparse

import torch

from relik.reader.data.relik_reader_sample import load_relik_reader_samples
from relik.reader.pytorch_modules.hf.modeling_relik import (
    RelikReaderConfig,
    RelikReaderREModel,
)
from relik.reader.pytorch_modules.triplet import RelikReaderForTripletExtraction
from relik.reader.utils.relation_matching_eval import StrongMatching
from relik.inference.data.objects import AnnotationType


def eval(model_path, data_path, is_eval, output_path=None):
    if model_path.endswith(".ckpt"):
        # if it is a lightning checkpoint we load the model state dict and the tokenizer from the config
        model_dict = torch.load(model_path)

        additional_special_symbols = model_dict["hyper_parameters"][
            "additional_special_symbols"
        ]
        from transformers import AutoTokenizer

        from relik.reader.utils.special_symbols import get_special_symbols_re

        special_symbols = get_special_symbols_re(additional_special_symbols - 1)
        tokenizer = AutoTokenizer.from_pretrained(
            model_dict["hyper_parameters"]["transformer_model"],
            additional_special_tokens=special_symbols,
            add_prefix_space=True,
        )
        config_model = RelikReaderConfig(
            model_dict["hyper_parameters"]["transformer_model"],
            len(special_symbols),
            training=False,
        )
        model = RelikReaderREModel(config_model)
        model_dict["state_dict"] = {
            k.replace("relik_reader_re_model.", ""): v
            for k, v in model_dict["state_dict"].items()
        }
        model.load_state_dict(model_dict["state_dict"], strict=False)
        reader = RelikReaderForTripletExtraction(
            model, training=False, device="cuda", tokenizer=tokenizer
        )
    else:
        # if it is a huggingface model we load the model directly. Note that it could even be a string from the hub
        model = RelikReaderREModel.from_pretrained(model_path, ignore_mismatched_sizes=True)
        reader = RelikReaderForTripletExtraction(
            model, training=False, device="cuda"
        )  # , dataset_kwargs={"use_nme": True}) if we want to use NME

    samples = list(load_relik_reader_samples(data_path))

    predicted_samples = reader.read(samples=samples, progress_bar=True, annotation_type=AnnotationType.WORD)
    if is_eval:
        strong_matching_metric = StrongMatching()
        predicted_samples = list(predicted_samples)
        for k, v in strong_matching_metric(predicted_samples).items():
            print(f"test_{k}", v)
    if output_path is not None:
        with open(output_path, "w") as f:
            for sample in predicted_samples:
                f.write(sample.to_jsons() + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/root/relik/experiments/relik_reader_re_small",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="/root/relik/data/re/test.jsonl",
    )
    parser.add_argument("--is-eval", action="store_true")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()
    eval(args.model_path, args.data_path, args.is_eval, args.output_path)


if __name__ == "__main__":
    main()
