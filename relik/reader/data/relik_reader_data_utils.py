from typing import List

import numpy as np
import torch


def flatten(lsts: List[list]) -> list:
    acc_lst = list()
    for lst in lsts:
        acc_lst.extend(lst)
    return acc_lst


def batchify(tensors: List[torch.Tensor], padding_value: int = 0) -> torch.Tensor:
    return torch.nn.utils.rnn.pad_sequence(
        tensors, batch_first=True, padding_value=padding_value
    )


def batchify_matrices(tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    x = max([t.shape[0] for t in tensors])
    y = max([t.shape[1] for t in tensors])
    out_matrix = torch.zeros((len(tensors), x, y))
    out_matrix += padding_value
    for i, tensor in enumerate(tensors):
        out_matrix[i][0 : tensor.shape[0], 0 : tensor.shape[1]] = tensor
    return out_matrix


def batchify_tensor(tensors: List[torch.Tensor], padding_value: int) -> torch.Tensor:
    x = max([t.shape[0] for t in tensors])
    y = max([t.shape[1] for t in tensors])
    rest = tensors[0].shape[2]
    out_matrix = torch.zeros((len(tensors), x, y, rest))
    out_matrix += padding_value
    for i, tensor in enumerate(tensors):
        out_matrix[i][0 : tensor.shape[0], 0 : tensor.shape[1], :] = tensor
    return out_matrix


def chunks(lst: list, chunk_size: int) -> List[list]:
    chunks_acc = list()
    for i in range(0, len(lst), chunk_size):
        chunks_acc.append(lst[i : i + chunk_size])
    return chunks_acc


def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = np.random.uniform(-noise_value, noise_value)
    return max(1, value + noise)
