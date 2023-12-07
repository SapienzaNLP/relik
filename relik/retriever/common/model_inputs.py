from __future__ import annotations

from collections import UserDict
from typing import Any, Union

import torch
from lightning.fabric.utilities import move_data_to_device

from relik.common.log import get_logger

logger = get_logger(__name__)


class ModelInputs(UserDict):
    """Model input dictionary wrapper."""

    def __getattr__(self, item: str):
        try:
            return self.data[item]
        except KeyError:
            raise AttributeError(f"`ModelInputs` has no attribute `{item}`")

    def __getitem__(self, item: str) -> Any:
        return self.data[item]

    def __getstate__(self):
        return {"data": self.data}

    def __setstate__(self, state):
        if "data" in state:
            self.data = state["data"]

    def keys(self):
        """A set-like object providing a view on D's keys."""
        return self.data.keys()

    def values(self):
        """An object providing a view on D's values."""
        return self.data.values()

    def items(self):
        """A set-like object providing a view on D's items."""
        return self.data.items()

    def to(self, device: Union[str, torch.device]) -> ModelInputs:
        """
        Send all tensors values to device.
        Args:
            device (`str` or `torch.device`): The device to put the tensors on.
        Returns:
            :class:`tokenizers.ModelInputs`: The same instance of :class:`~tokenizers.ModelInputs`
            after modification.
        """
        self.data = move_data_to_device(self.data, device)
        return self
