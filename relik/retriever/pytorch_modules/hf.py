from typing import Tuple, Union

import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel


class GoldenRetrieverConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout


class GoldenRetrieverModel(BertModel):
    config_class = GoldenRetrieverConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.layer_norm_layer = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )

    def forward(
        self, **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        attention_mask = kwargs.get("attention_mask", None)
        model_outputs = super().forward(**kwargs)
        if attention_mask is None:
            pooler_output = model_outputs.pooler_output
        else:
            token_embeddings = model_outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            pooler_output = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

            pooler_output = self.layer_norm_layer(pooler_output)

        if not kwargs.get("return_dict", True):
            return (model_outputs[0], pooler_output) + model_outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=model_outputs.last_hidden_state,
            pooler_output=pooler_output,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            cross_attentions=model_outputs.cross_attentions,
        )
