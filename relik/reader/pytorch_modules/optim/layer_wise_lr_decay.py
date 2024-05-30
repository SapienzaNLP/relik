import collections
from typing import Dict, List, Tuple, Union

import torch
from torch.optim import AdamW
import transformers

class LayerWiseLRDecayOptimizer:
    def __init__(
        self,
        lr: Union[float, List[float]],
        warmup_steps: int,
        total_steps: int,
        weight_decay: float,
        lr_decay: float,
        no_decay_params: List[str],
        total_reset: int,
        other_lr_params: List[str] = [],
    ):
        # Simplify lr handling
        self.lr = lr if isinstance(lr, float) else lr[0]
        self.lr2 = None if isinstance(lr, float) else lr[1]
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.no_decay_params = no_decay_params
        self.other_lr_params = other_lr_params
        self.total_reset = total_reset

    def group_layers(self, module) -> dict:
        grouped_layers = collections.defaultdict(list)
        module_named_parameters = list(module.named_parameters())
        for ln, lp in module_named_parameters:
            if "embeddings" in ln:
                grouped_layers["embeddings"].append((ln, lp))
            elif "encoder.layer" in ln:
                layer_num = ln.split("transformer_model.encoder.layer.")[-1]
                layer_num = layer_num[0 : layer_num.index(".")]
                grouped_layers[layer_num].append((ln, lp))
            else:
                grouped_layers["head"].append((ln, lp))

        depth = len(grouped_layers) - 1
        final_dict = dict()
        for key, value in grouped_layers.items():
            if key == "head":
                final_dict[0] = value
            elif key == "embeddings":
                final_dict[depth] = value
            else:
                # -1 because layer number starts from zero
                final_dict[depth - int(key) - 1] = value

        assert len(module_named_parameters) == sum(
            len(v) for _, v in final_dict.items()
        )

        return final_dict

    def group_params(self, module: torch.nn.Module) -> List[Dict]:
        optimizer_grouped_params = []
        for inverse_depth, layer in self.group_layers(module).items():
            layer_lr = self.lr * (self.lr_decay**inverse_depth)
            for ln, lp in layer:
                lr = self.lr2 if any(nd in ln for nd in self.other_lr_params) else layer_lr
                weight_decay = 0.0 if any(nd in ln for nd in self.no_decay_params) else self.weight_decay

                optimizer_grouped_params.append({"params": [lp], "lr": lr, "weight_decay": weight_decay})

        return optimizer_grouped_params

    def __call__(self, module: torch.nn.Module):
        optimizer_grouped_parameters = self.group_params(module)
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
        scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            self.warmup_steps,
            self.total_steps,
            num_cycles=self.total_reset,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

# import collections
# from typing import List

# import torch
# import transformers
# from torch.optim import AdamW


# class LayerWiseLRDecayOptimizer:
#     def __init__(
#         self,
#         lr: float,
#         warmup_steps: int,
#         total_steps: int,
#         weight_decay: float,
#         lr_decay: float,
#         no_decay_params: List[str],
#         total_reset: int,
#     ):
#         self.lr = lr
#         self.warmup_steps = warmup_steps
#         self.total_steps = total_steps
#         self.weight_decay = weight_decay
#         self.lr_decay = lr_decay
#         self.no_decay_params = no_decay_params
#         self.total_reset = total_reset

#     def group_layers(self, module) -> dict:
#         grouped_layers = collections.defaultdict(list)
#         module_named_parameters = list(module.named_parameters())
#         for ln, lp in module_named_parameters:
#             if "embeddings" in ln:
#                 grouped_layers["embeddings"].append((ln, lp))
#             elif "encoder.layer" in ln:
#                 layer_num = ln.split("transformer_model.encoder.layer.")[-1]
#                 layer_num = layer_num[0 : layer_num.index(".")]
#                 grouped_layers[layer_num].append((ln, lp))
#             else:
#                 grouped_layers["head"].append((ln, lp))

#         depth = len(grouped_layers) - 1
#         final_dict = dict()
#         for key, value in grouped_layers.items():
#             if key == "head":
#                 final_dict[0] = value
#             elif key == "embeddings":
#                 final_dict[depth] = value
#             else:
#                 # -1 because layer number starts from zero
#                 final_dict[depth - int(key) - 1] = value

#         assert len(module_named_parameters) == sum(
#             len(v) for _, v in final_dict.items()
#         )

#         return final_dict

#     def group_params(self, module) -> list:
#         optimizer_grouped_params = []
#         for inverse_depth, layer in self.group_layers(module).items():
#             layer_lr = self.lr * (self.lr_decay**inverse_depth)
#             layer_wd_params = {
#                 "params": [
#                     lp
#                     for ln, lp in layer
#                     if not any(nd in ln for nd in self.no_decay_params)
#                 ],
#                 "weight_decay": self.weight_decay,
#                 "lr": layer_lr,
#             }
#             layer_no_wd_params = {
#                 "params": [
#                     lp
#                     for ln, lp in layer
#                     if any(nd in ln for nd in self.no_decay_params)
#                 ],
#                 "weight_decay": 0,
#                 "lr": layer_lr,
#             }

#             if len(layer_wd_params) != 0:
#                 optimizer_grouped_params.append(layer_wd_params)
#             if len(layer_no_wd_params) != 0:
#                 optimizer_grouped_params.append(layer_no_wd_params)

#         return optimizer_grouped_params

#     def __call__(self, module: torch.nn.Module):
#         optimizer_grouped_parameters = self.group_params(module)
#         optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr)
#         scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
#             optimizer,
#             self.warmup_steps,
#             self.total_steps,
#             num_cycles=self.total_reset,
#         )
#         return {
#             "optimizer": optimizer,
#             "lr_scheduler": {
#                 "scheduler": scheduler,
#                 "interval": "step",
#                 "frequency": 1,
#             },
#         }
