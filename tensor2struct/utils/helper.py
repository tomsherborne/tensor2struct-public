import argparse
import shutil
import sys
import os
import re
import json
import time

import torch
import logging
import torch.distributed as dist

def format_size(size: int) -> str:
    """
    Format a size (in bytes) for humans.
    """
    GBs = size / (1024 * 1024 * 1024)
    if GBs >= 10:
        return f"{int(round(GBs, 0))}G"
    if GBs >= 1:
        return f"{round(GBs, 1):.1f}G"
    MBs = size / (1024 * 1024)
    if MBs >= 10:
        return f"{int(round(MBs, 0))}M"
    if MBs >= 1:
        return f"{round(MBs, 1):.1f}M"
    KBs = size / 1024
    if KBs >= 10:
        return f"{int(round(KBs, 0))}K"
    if KBs >= 1:
        return f"{round(KBs, 1):.1f}K"
    return f"{size}B"


def peak_gpu_memory():
    """
    Get the peak GPU memory usage in bytes by device.

    # Returns

    `Dict[int, int]`
        Keys are device ids as integers.
        Values are memory usage as integers in bytes.
        Returns an empty `dict` if GPUs are not available.
    """
    if not torch.cuda.is_available():
        return {}

    if dist.is_available() and dist.is_initialized():
        # If the backend is not 'nccl', we're training on CPU.
        if dist.get_backend() != "nccl":
            return {}

        device = torch.cuda.current_device()
        global_rank = dist.get_rank()
        world_size = dist.get_world_size()
        peak_bytes = torch.cuda.max_memory_allocated(device)
        peak_bytes_tensor = torch.tensor([global_rank, peak_bytes], device=device)
        # All of these tensors will be gathered into this list.
        gather_results = [torch.tensor([0, 0], device=device) for _ in range(world_size)]

        dist.all_gather(gather_results, peak_bytes_tensor)

        results_dict = {}
        for peak_bytes_tensor in gather_results:
            results_dict[int(peak_bytes_tensor[0])] = int(peak_bytes_tensor[1])

        return results_dict
    else:
        return {0: torch.cuda.max_memory_allocated()}
