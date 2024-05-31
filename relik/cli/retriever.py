import hydra
import typer

from relik.cli.utils import resolve_config
from relik.common.log import get_logger, print_relik_text_art
from relik.retriever.trainer import train as retriever_train

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)


@app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
def train():
    print_relik_text_art()
    config = resolve_config("retriever")
    retriever_train(config)
