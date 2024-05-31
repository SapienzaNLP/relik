from typing import Any, Optional

import lightning
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from relik.reader.pytorch_modules.span import RelikReaderForSpanExtraction


class RelikReaderPLModule(lightning.LightningModule):
    def __init__(
        self,
        cfg: dict,
        transformer_model: str,
        additional_special_symbols: int,
        num_layers: Optional[int] = None,
        activation: str = "gelu",
        linears_hidden_size: Optional[int] = 512,
        use_last_k_layers: int = 1,
        training: bool = False,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.relik_reader_core_model = RelikReaderForSpanExtraction(
            transformer_model,
            additional_special_symbols,
            num_layers,
            activation,
            linears_hidden_size,
            use_last_k_layers,
            training=training,
        )
        self.optimizer_factory = None

    def training_step(self, batch: dict, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        relik_output = self.relik_reader_core_model(**batch)
        self.log("train-loss", relik_output["loss"])
        return relik_output["loss"]

    def validation_step(
        self, batch: dict, *args: Any, **kwargs: Any
    ) -> Optional[STEP_OUTPUT]:
        return

    def set_optimizer_factory(self, optimizer_factory) -> None:
        self.optimizer_factory = optimizer_factory

    def configure_optimizers(self) -> OptimizerLRScheduler:
        return self.optimizer_factory(self.relik_reader_core_model)
