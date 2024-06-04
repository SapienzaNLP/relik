import hydra
import typer

from relik.cli.utils import resolve_config
from relik.common.log import get_logger, print_relik_text_art
from relik.reader.trainer.train import train as reader_train

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def train():
    print_relik_text_art()
    config, config_dir = resolve_config("reader")
    hydra.initialize_config_dir(config_dir=str(config_dir), version_base="1.3")
    reader_train(config)
