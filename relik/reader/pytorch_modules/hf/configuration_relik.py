from typing import Optional

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig


class RelikReaderConfig(PretrainedConfig):
    model_type = "relik-reader"

    def __init__(
        self,
        transformer_model: str = "microsoft/deberta-v3-base",
        additional_special_symbols: int = 101,
        additional_special_symbols_types: Optional[int] = 0,
        num_layers: Optional[int] = None,
        activation: str = "gelu",
        linears_hidden_size: Optional[int] = 512,
        use_last_k_layers: int = 1,
        entity_type_loss: bool = False,
        add_entity_embedding: bool = None,
        training: bool = False,
        default_reader_class: Optional[str] = None,
        **kwargs
    ) -> None:
        # TODO: add name_or_path to kwargs
        self.transformer_model = transformer_model
        self.additional_special_symbols = additional_special_symbols
        self.additional_special_symbols_types = additional_special_symbols_types
        self.num_layers = num_layers
        self.activation = activation
        self.linears_hidden_size = linears_hidden_size
        self.use_last_k_layers = use_last_k_layers
        self.entity_type_loss = entity_type_loss
        self.add_entity_embedding = (
            True
            if add_entity_embedding is None and entity_type_loss
            else add_entity_embedding
        )
        self.training = training
        self.default_reader_class = default_reader_class
        super().__init__(**kwargs)


AutoConfig.register("relik-reader", RelikReaderConfig)
