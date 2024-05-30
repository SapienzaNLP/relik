import argparse
from pprint import pprint
from typing import Optional

from relik.reader.pytorch_modules.span import RelikReaderForSpanExtraction
from relik.reader.utils.strong_matching_eval import StrongMatching
from relik.reader.data.relik_reader_sample import load_relik_reader_samples

def predict(
    model_path: str,
    dataset_path: str,
    token_batch_size: int,
    is_eval: bool,
    output_path: Optional[str],
) -> None:
    relik_reader = RelikReaderForSpanExtraction(model_path, dataset_kwargs={"use_nme": True}, device="cuda")
    samples = list(load_relik_reader_samples(dataset_path))
    predicted_samples = relik_reader.read(
        samples=samples, token_batch_size=token_batch_size, progress_bar=True
    )
    if is_eval:
        eval_dict = StrongMatching()(predicted_samples)
        pprint(eval_dict)
    if output_path is not None:
        with open(output_path, "w") as f:
            for sample in predicted_samples:
                f.write(sample.to_jsons() + "\n")


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        required=True,
    )
    parser.add_argument("--dataset-path", "-i", required=True)
    parser.add_argument("--is-eval", action="store_true")
    parser.add_argument(
        "--output-path",
        "-o",
    )
    parser.add_argument("--token-batch-size", default=4096)
    return parser.parse_args()


def main():
    args = parse_arg()
    predict(
        args.model_path,
        args.dataset_path,
        token_batch_size=args.token_batch_size,
        is_eval=args.is_eval,
        output_path=args.output_path,
    )


if __name__ == "__main__":
    main()
