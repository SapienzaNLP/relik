import hydra
import typer

from relik.cli.utils import resolve_config
from relik.common.log import get_logger, print_relik_text_art
from relik.cli import reader, retriever

logger = get_logger(__name__)

app = typer.Typer(no_args_is_help=True, pretty_exceptions_show_locals=False)
app.add_typer(reader.app, name="reader")
app.add_typer(retriever.app, name="retriever")

# @app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# def test():
#     config = resolve_config()
#     trainer: Trainer = hydra.utils.instantiate(config, _recursive_=False)
#     trainer.test()


# @app.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# def train():
#     print_text_art()
#     config = resolve_config()
#     trainer = Trainer(**config)
#     trainer.train()


if __name__ == "__main__":
    app()
