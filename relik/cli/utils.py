import sys
from pathlib import Path

import typer
from hydra import compose, initialize, initialize_config_dir
from omegaconf import OmegaConf

from relik.common.log import get_logger

logger = get_logger(__name__)


def resolve_config(type: str | None = None) -> OmegaConf:
    """
    Resolve the config file and return the OmegaConf object.

    Args:
        config_path (`str`):
            The path to the config file.

    Returns:
        `OmegaConf`:
            The OmegaConf object.
    """
    # first arg is the entry point
    # second arg is the command
    # third arg is the subcommand
    # fourth arg is the config path/name
    # fifth arg is the overrides
    _, _, _, config_path, *overrides = sys.argv
    config_path = Path(config_path)
    # TODO: do checks
    # if not config_path.exists():
    #     raise ValueError(f"File {config_path} does not exist!")
    # get path and name
    config_dir, config_name = config_path.parent, config_path.stem
    # logger.debug(f"config_path: {config_path}")
    # logger.debug(f"config_name: {config_name}")
    # check if config_path is absolute or relative
    # if config_path.is_absolute():
    #     context = initialize_config_dir(config_dir=str(config_path), version_base="1.3")
    # else:
    if not config_dir.is_absolute():
        base_path = Path(__file__).parent.parent
        if type == "reader":
            config_dir = base_path / "reader" / "conf"
        elif type == "retriever":
            config_dir = base_path / "retriever" / "conf"
        else:
            raise ValueError(
                "Please provide the type (`reader` or `retriever`) or provide an absolute path."
            )
    logger.debug(f"config_dir: {config_dir}")
    # logger.debug(f"config_name: {config_name}")

    # print(OmegaConf.load(config_dir / f"{config_name}.yaml"))

    # with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
    #     cfg = compose(config_name=config_name, overrides=overrides)

    return config_dir, config_name, overrides


def int_or_str_typer(value: str) -> int | None:
    """
    Converts a string value to an integer or None.

    Args:
        value (str): The string value to be converted.

    Returns:
        int | None: The converted integer value or None if the input is "None".
    """
    if value == "None":
        return None
    return int(value)
