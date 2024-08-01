import contextlib
import tempfile

import torch
import transformers as tr


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
