import contextlib
import tempfile

import torch
import transformers as tr

from relik.common.utils import is_package_available

# check if ORT is available
if is_package_available("onnxruntime"):
    from optimum.onnxruntime import (
        ORTModel,
        ORTModelForCustomTasks,
        ORTModelForSequenceClassification,
        ORTOptimizer,
    )
    from optimum.onnxruntime.configuration import AutoOptimizationConfig

# from relik.retriever.pytorch_modules import PRECISION_MAP


def get_autocast_context(
    device: str | torch.device, precision: str
) -> contextlib.AbstractContextManager:
    # fucking autocast only wants pure strings like 'cpu' or 'cuda'
    # we need to convert the model device to that
    device_type_for_autocast = str(device).split(":")[0]

    from relik.retriever.pytorch_modules import PRECISION_MAP

    # autocast doesn't work with CPU and stuff different from bfloat16
    autocast_manager = (
        contextlib.nullcontext()
        if device_type_for_autocast in ["cpu", "mps"]
        and PRECISION_MAP[precision] != torch.bfloat16
        else (
            torch.autocast(
                device_type=device_type_for_autocast,
                dtype=PRECISION_MAP[precision],
            )
        )
    )
    return autocast_manager


# def load_ort_optimized_hf_model(
#     hf_model: tr.PreTrainedModel,
#     provider: str = "CPUExecutionProvider",
#     ort_model_type: callable = "ORTModelForCustomTasks",
# ) -> ORTModel:
#     """
#     Load an optimized ONNX Runtime HF model.
#
#     Args:
#         hf_model (`tr.PreTrainedModel`):
#             The HF model to optimize.
#         provider (`str`, optional):
#             The ONNX Runtime provider to use. Defaults to "CPUExecutionProvider".
#
#     Returns:
#         `ORTModel`: The optimized HF model.
#     """
#     if isinstance(hf_model, ORTModel):
#         return hf_model
#     temp_dir = tempfile.mkdtemp()
#     hf_model.save_pretrained(temp_dir)
#     ort_model = ort_model_type.from_pretrained(
#         temp_dir, export=True, provider=provider, use_io_binding=True
#     )
#     if is_package_available("onnxruntime"):
#         optimizer = ORTOptimizer.from_pretrained(ort_model)
#         optimization_config = AutoOptimizationConfig.O4()
#         optimizer.optimize(save_dir=temp_dir, optimization_config=optimization_config)
#         ort_model = ort_model_type.from_pretrained(
#             temp_dir,
#             export=True,
#             provider=provider,
#             use_io_binding=bool(provider == "CUDAExecutionProvider"),
#         )
#         return ort_model
#     else:
#         raise ValueError("onnxruntime is not installed. Please install Ray with `pip install relik[serve]`.")
