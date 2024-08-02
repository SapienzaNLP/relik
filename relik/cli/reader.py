import sys

import hydra
import typer
from omegaconf import OmegaConf

from relik.cli.utils import resolve_config
from relik.common.log import get_logger, print_relik_text_art
from relik.reader.trainer.train import train as reader_train
from relik.reader.trainer.train_cie import train as reader_train_cie

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def train():
    """
    Trains the reader model.

    This function prints the Relik text art, resolves the configuration file path,
    and then calls the `_reader_train` function to train the reader model.

    Args:
        None

    Returns:
        None
    """
    print_relik_text_art()
    config_dir, config_name, overrides = resolve_config("reader")

    @hydra.main(
        config_path=str(config_dir),
        config_name=str(config_name),
        version_base="1.3",
    )
    def _reader_train(conf):
        reader_train(conf)

    # clean sys.argv for hydra
    sys.argv = sys.argv[:1]
    # add the overrides to sys.argv
    sys.argv.extend(overrides)

    _reader_train()


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def train_cie():
    """
    Trains the reader model.

    This function prints the Relik text art, resolves the configuration file path,
    and then calls the `_reader_train` function to train the reader model.

    Args:
        None

    Returns:
        None
    """
    print_relik_text_art()
    config_dir, config_name, overrides = resolve_config("reader")

    @hydra.main(
        config_path=str(config_dir),
        config_name=str(config_name),
        version_base="1.3",
    )
    def _reader_train_cie(conf):
        reader_train_cie(conf)

    # clean sys.argv for hydra
    sys.argv = sys.argv[:1]
    # add the overrides to sys.argv
    sys.argv.extend(overrides)

    _reader_train_cie()
