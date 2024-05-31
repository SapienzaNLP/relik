import argparse
import os
from typing import Tuple

import omegaconf
import torch

from relik.common.utils import from_cache
from relik.reader.lightning_modules.relik_reader_pl_module import RelikReaderPLModule

CKPT_FILE_NAME = "model.ckpt"
CONFIG_FILE_NAME = "cfg.yaml"


def convert_pl_module(pl_module_ckpt_path: str, output_dir: str) -> None:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        print(f"{output_dir} already exists, aborting operation")
        exit(1)

    relik_pl_module: RelikReaderPLModule = RelikReaderPLModule.load_from_checkpoint(
        pl_module_ckpt_path
    )
    torch.save(
        relik_pl_module.relik_reader_core_model, f"{output_dir}/{CKPT_FILE_NAME}"
    )
    with open(f"{output_dir}/{CONFIG_FILE_NAME}", "w") as f:
        omegaconf.OmegaConf.save(
            omegaconf.OmegaConf.create(relik_pl_module.hparams["cfg"]), f
        )


def load_model_and_conf(
    model_dir_path: str,
) -> Tuple[torch.nn.Module, omegaconf.DictConfig]:
    # TODO: quick workaround to load the model from HF hub
    model_dir = from_cache(
        model_dir_path,
        filenames=[CKPT_FILE_NAME, CONFIG_FILE_NAME],
        cache_dir=None,
        force_download=False,
    )

    ckpt_path = f"{model_dir}/{CKPT_FILE_NAME}"
    model = torch.load(ckpt_path, map_location=torch.device("cpu"))

    model_cfg_path = f"{model_dir}/{CONFIG_FILE_NAME}"
    model_conf = omegaconf.OmegaConf.load(model_cfg_path)
    return model, model_conf


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        help="Path to the pytorch lightning ckpt you want to convert.",
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        help="The output dir to store the bare models and the config.",
        required=True,
    )
    return parser.parse_args()


def main():
    args = parse_arg()
    convert_pl_module(args.ckpt, args.output_dir)


if __name__ == "__main__":
    main()
