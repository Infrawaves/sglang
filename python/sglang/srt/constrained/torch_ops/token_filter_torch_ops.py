# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Token filters for CPU masks and accelerator torch fallbacks.

CPU masks use a shared NumPy view to avoid per-token tensor operations.
"""

import ctypes
from typing import List

import torch


def set_token_filter_torch(
    vocab_mask: torch.Tensor,
    token_ids: List[int],
    batch_idx: int,
    is_allowed: bool = True,
    reset_vocab_mask: bool = True,
):
    if vocab_mask.device.type == "cpu":
        row = vocab_mask.numpy()[batch_idx]
        if reset_vocab_mask:
            row.fill(-1 if not is_allowed else 0)
        for token_id in token_ids:
            element_idx, bit_idx = divmod(token_id, 32)
            bitmask = ctypes.c_int32(1 << bit_idx).value
            value = int(row[element_idx])
            row[element_idx] = value | bitmask if is_allowed else value & ~bitmask
        return

    if reset_vocab_mask:
        vocab_mask[batch_idx].fill_(-1 if (not is_allowed) else 0)

    if not token_ids:
        return

    # Aggregate bit masks per int32 element to handle duplicate indices.
    aggregated: dict[int, int] = {}
    for token_id in token_ids:
        element_idx = token_id // 32
        bit_idx = token_id % 32
        aggregated[element_idx] = aggregated.get(element_idx, 0) | (1 << bit_idx)

    row = vocab_mask[batch_idx]
    element_indices = torch.tensor(
        list(aggregated.keys()), dtype=torch.long, device=row.device
    )
    bitmasks = torch.tensor(
        [
            ctypes.c_int32(mask if is_allowed else ~mask).value
            for mask in aggregated.values()
        ],
        dtype=row.dtype,
        device=row.device,
    )

    if is_allowed:
        row[element_indices] = torch.bitwise_or(row[element_indices], bitmasks)
    else:
        row[element_indices] = torch.bitwise_and(row[element_indices], bitmasks)
