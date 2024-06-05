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
    sample_index_file: Union[str, os.PathLike],
):

    logger.info(f"Loading sample index from {sample_index_file}")
    with open(sample_index_file) as f:
        sample_index = json.load(f)
        # get all unique values
        sample_index = set(
            [item for sublist in sample_index.values() for item in sublist]
        )

    output_file_path = Path(output_file)
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Loading data from {input_file} and sampling")
    logger.info(f"Saving sampled data to {output_file}")
    with open(input_file) as f, open(output_file, "w") as f_out:
        for i, line in tqdm(enumerate(f)):
            if int(i) in sample_index:
                try:
                    f_out.write(json.dumps(json.loads(line)) + "\n")
                except json.JSONDecodeError:
                    logger.error(f"Error parsing line {i}")
                    continue


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("input_file", type=str)
    arg_parser.add_argument("output_file", type=str)
    arg_parser.add_argument("sample_index_file", type=str)

    sample(**vars(arg_parser.parse_args()))
