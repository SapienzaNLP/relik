import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import numpy as np
from tqdm import tqdm

from relik.common.log import get_logger

logger = get_logger()


def sample(
    input_file: Union[str, os.PathLike],
    output_file: Union[str, os.PathLike],
    n_samples: int = 5,
    seed: int = 42,
):
    documents = defaultdict(list)

    logger.info(f"Loading data from {input_file}")
    with open(input_file) as f:
        for i, line in tqdm(enumerate(f)):
            try:
                sample = json.loads(line)
                # data.append(sample)
                labels = [l[-1] for l in sample["window_labels"]]
                for label in labels:
                    documents[label].append(i)
            except json.JSONDecodeError:
                logger.error(f"Error parsing line {i}")
                continue

    logger.info("Sampling data")
    # Random sample from in-distribution documents
    np.random.seed(seed)
    documents = {
        k: np.random.choice(v, min(len(v), n_samples), replace=False).tolist()
        for k, v in tqdm(documents.items())
    }

    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving sampled data to {output_file}")
    with open(output_file, "w") as f:
        json.dump(documents, f, indent=2)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("--n_samples", type=int, required=False, default=5)

    sample(**vars(arg_parser.parse_args()))
