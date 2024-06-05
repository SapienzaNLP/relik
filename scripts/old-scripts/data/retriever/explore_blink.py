import argparse
import json


if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser()
    # arg_parser.add_argument("--question_encoder_name_or_path", type=str, required=True)
    # arg_parser.add_argument("--document_path", type=str, required=True)
    # arg_parser.add_argument("--passage_encoder_name_or_path", type=str)
    # arg_parser.add_argument(
    #     "--indexer_class",
    #     type=str,
    #     default="relik.retriever.indexers.inmemory.InMemoryDocumentIndex",
    # )
    # arg_parser.add_argument("--document_file_type", type=str, default="jsonl")
    # arg_parser.add_argument("--output_folder", type=str, required=True)
    # arg_parser.add_argument("--batch_size", type=int, default=128)
    # arg_parser.add_argument("--passage_max_length", type=int, default=64)
    # arg_parser.add_argument("--device", type=str, default="cuda")
    # arg_parser.add_argument("--index_device", type=str, default="cpu")
    # arg_parser.add_argument("--precision", type=str, default="fp32")

    # build_index(**vars(arg_parser.parse_args()))

    with open("/media/data/EL/blink/window_32_tokens/random_1M/dpr-like/first_1M.jsonl") as f:
        data = [json.loads(line) for line in f]
    
    