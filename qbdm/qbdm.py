import torch
import numpy as np
from pybdm import BDM
from concurrent.futures import ProcessPoolExecutor
import torch.nn as nn

# Global BDM instance for workers
bdm_instance = BDM(ndim=2, nsymbols=2)

def bdm_batch_worker(data_list):
    """Processes a batch of binary planes in a single worker call to reduce overhead."""
    results = []
    for data in data_list:
        try:
            results.append(float(bdm_instance.bdm(data)))
        except Exception:
            results.append(0.0)
    return results

def get_bitplanes(weight_tensor, num_planes, robust=False, percentile=99.9):
    """Vectorized GPU-based bit-plane extraction with optional robust normalization."""
    if weight_tensor.dim() > 2:
        weight_tensor = weight_tensor.flatten(1)
    h, w = weight_tensor.shape
    if h < 4 or w < 4:
        return []

    if robust:
        # Compute quantiles to clamp extreme outliers
        flat_w = weight_tensor.view(-1).float()
        q = (1.0 - percentile / 100.0) / 2.0
        w_min = torch.quantile(flat_w, q)
        w_max = torch.quantile(flat_w, 1.0 - q)
        weight_tensor = torch.clamp(weight_tensor, w_min, w_max)
    else:
        w_min, w_max = weight_tensor.min(), weight_tensor.max()

    if w_max == w_min:
        return [np.zeros((h, w), dtype=np.int8)] * num_planes

    scaled = ((weight_tensor - w_min) / (w_max - w_min + 1e-8) * (2**num_planes - 1)).to(torch.uint8)
    planes = []
    for i in range(num_planes):
        plane_np = ((scaled >> i) & 1).cpu().numpy().astype(np.int8)
        planes.append(plane_np)
    return planes

def get_tiled_manifold(weight_tensor, num_planes, robust=False, percentile=99.9):
    """Constructs a 2D tiled manifold to capture inter-plane coupling."""
    planes = get_bitplanes(weight_tensor, num_planes, robust, percentile)
    if not planes:
        return None

    h, w = planes[0].shape
    if num_planes == 4:
        tr, tc = 2, 2
    elif num_planes == 8:
        tr, tc = 4, 2
    elif num_planes == 2:
        tr, tc = 2, 1
    else:
        tr, tc = num_planes, 1

    tiled = np.zeros((h * tr, w * tc), dtype=np.int8)
    for i in range(tr):
        for j in range(tc):
            bit_idx = i * tc + j
            if bit_idx < num_planes:
                tiled[i::tr, j::tc] = planes[bit_idx]
    return tiled


def measure_complexity(input_obj, bit_depths=[8], max_workers=8, robust=False, percentile=99.9, tiled=False):
    """Aggregates complexity scores for independent and optionally tiled manifolds."""
    binary_tasks = []
    multi_tasks = {bd: [] for bd in bit_depths}
    tiled_tasks = {bd: [] for bd in bit_depths} if tiled else {}

    # Identify tensors to process: either from a model or a direct tensor input
    if isinstance(input_obj, nn.Module):
        tensors = [p.data for name, p in input_obj.named_parameters() if p.requires_grad and p.dim() >= 2]
    elif isinstance(input_obj, torch.Tensor):
        tensors = [input_obj] if input_obj.dim() >= 2 else []
    else:
        raise ValueError("Input must be a torch.nn.Module or a multi-dimensional torch.Tensor")

    for p_data in tensors:
        binary_tasks.append((p_data.data > 0).cpu().numpy().astype(np.int8))
        for bd in bit_depths:
            if tiled:
                tm = get_tiled_manifold(p_data.data, bd, robust=robust, percentile=percentile)
                if tm is not None: tiled_tasks[bd].append(tm)

            planes = get_bitplanes(p_data.data, bd, robust=robust, percentile=percentile)
            multi_tasks[bd].extend(planes)

    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        bin_chunks = list(chunk_list(binary_tasks, max(1, len(binary_tasks) // max_workers)))
        bin_results = list(executor.map(bdm_batch_worker, bin_chunks))
        total_bin = sum(sum(res) for res in bin_results)

        total_multi = {}
        total_tiled = {} if tiled else None

        for bd in bit_depths:
            m_tasks = multi_tasks[bd]
            m_chunks = list(chunk_list(m_tasks, max(1, len(m_tasks) // (max_workers * 2))))
            m_results = list(executor.map(bdm_batch_worker, m_chunks))
            total_multi[bd] = sum(sum(res) for res in m_results)

            if tiled:
                t_tasks = tiled_tasks[bd]
                t_chunks = list(chunk_list(t_tasks, max(1, len(t_tasks) // max_workers)))
                t_results = list(executor.map(bdm_batch_worker, t_chunks))
                total_tiled[bd] = sum(sum(res) for res in t_results)

        return float(total_bin), total_multi, total_tiled



