import torch
import numpy as np
from pybdm import BDM
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import gzip
import lzma
import pdb
import torch.nn as nn

# Global BDM instance for workers
bdm_instance = BDM(ndim=2, nsymbols=2)

def _eligible_weight_params(model):
    """Yields the same trainable, 2D+ weight tensors that measure_complexity() collects
    (see its tensor-selection logic below), handling parametrized (e.g. QAT) modules by
    yielding the underlying raw parameter rather than the computed/quantized view."""
    for module in model.modules():
        if not hasattr(module, 'weight'):
            continue
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            w = module.parametrizations.weight.original
        else:
            w = module.weight
        if isinstance(w, torch.nn.Parameter) and w.requires_grad and w.dim() >= 2:
            yield w

@contextmanager
def shuffled_weights(model):
    """Randomly permutes each eligible weight tensor's values in place for the duration of
    the `with` block, then restores the originals.

    This preserves each tensor's exact value distribution (so the quantizer's dynamic range,
    and zeroth-order compressibility, are unchanged) while destroying any learned spatial/
    structural pattern. Comparing a trained model's complexity/compression against its own
    shuffled self isolates genuine learned structure from whatever is attributable to the
    value distribution alone -- a "true structure" baseline, as opposed to comparing against
    a separately-initialized random model.

    Usage:
        with shuffled_weights(model):
            s_bin, s_multi, _ = measure_complexity(model, ...)
            s_comp = measure_compression(model)
    """
    originals = []
    with torch.no_grad():
        for w in _eligible_weight_params(model):
            flat = w.data.view(-1)
            originals.append((w, flat.clone()))
            perm = torch.randperm(flat.numel(), device=flat.device)
            flat.copy_(flat[perm])
    try:
        yield model
    finally:
        with torch.no_grad():
            for w, original in originals:
                w.data.view(-1).copy_(original)

def bdm_batch_worker(data_list):
    """Processes a batch of binary planes in a single worker call to reduce overhead."""
    results = []
    for data in data_list:
        try:
            results.append(float(bdm_instance.bdm(data)))
        except Exception:
            results.append(0.0)
    return results

def get_bitplanes(weight_tensor, num_planes, robust=False, percentile=99.9, epsilon=1e-8):
    """Vectorized GPU-based bit-plane extraction with optional robust normalization."""
    if weight_tensor.dim() > 2:
        weight_tensor = weight_tensor.flatten(1)
    h, w = weight_tensor.shape
    if h < 4 or w < 4:
        return []

    # Determine dynamic range boundaries for the quantizer
    if robust:
        flat_w = weight_tensor.view(-1).float()
        q = (1.0 - percentile / 100.0) / 2.0
        w_min = torch.quantile(flat_w, q)
        w_max = torch.quantile(flat_w, 1.0 - q)
    else:
        w_min, w_max = weight_tensor.min(), weight_tensor.max()

    # Calculate the quantization step size (Delta)
    delta = (w_max - w_min) / (2**num_planes - 1)

    # Map to integer space using standard uniform rounding and saturation (clamping)
    # Adding a small epsilon to delta to prevent division by zero
    if (w_max - w_min) < epsilon:
        quantized = [np.zeros((h, w), dtype=np.int8)] * num_planes
    else:
        quantized = torch.clamp(
            torch.round((weight_tensor - w_min) / (delta)),
            0, 2**num_planes - 1
        )#.to(torch.uint8)

    planes = []
    for i in range(num_planes):
        plane = torch.div(quantized, 2**i, rounding_mode='floor') % 2
        #plane_np = ((quantized >> i) & 1).cpu().numpy().astype(np.int8)
        plane_np = plane.cpu().numpy().astype(np.int8)
        planes.append(plane_np)
    return planes

def measure_compression(model):
    """Calculates global compression savings by operating on raw weight tensors directly."""
    results = {'gzip_raw': 0.0, 'lzma_raw': 0.0, 'raw_bytes': 0.0}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                weights_np = p.data.cpu().numpy()
                byte_stream = weights_np.tobytes()
                results['raw_bytes'] += len(byte_stream)
                results['gzip_raw'] += len(gzip.compress(byte_stream))
                results['lzma_raw'] += len(lzma.compress(byte_stream))

    savings = {
        'gzip': (1 - results['gzip_raw'] / results['raw_bytes']) * 100 if results['raw_bytes'] > 0 else 0,
        'lzma': (1 - results['lzma_raw'] / results['raw_bytes']) * 100 if results['raw_bytes'] > 0 else 0
    }
    return savings

def measure_bitplane_compression(model, bit_depths=[8]):
    """Calculates additive compression savings by compressing each bit-plane independently."""
    results = {bd: {'gzip': 0.0, 'lzma': 0.0, 'raw': 0.0} for bd in bit_depths}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad and p.dim() >= 2:
                for bd in bit_depths:
                    planes = get_bitplanes(p.data, bd)
                    for plane in planes:
                        packed = np.packbits(plane.flatten())
                        results[bd]['raw'] += packed.nbytes
                        results[bd]['gzip'] += len(gzip.compress(packed))
                        results[bd]['lzma'] += len(lzma.compress(packed))

    savings = {bd: {
        'gzip': (1 - results[bd]['gzip'] / results[bd]['raw']) * 100 if results[bd]['raw'] > 0 else 0,
        'lzma': (1 - results[bd]['lzma'] / results[bd]['raw']) * 100 if results[bd]['raw'] > 0 else 0
    } for bd in bit_depths}
    return savings

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


def measure_per_plane_complexity(model, bit_depth=8, max_workers=8, robust=False, percentile=99.9, tiled=False):
    """Aggregates complexity scores for individual bitplanes."""

    for name, module in model.named_modules():
        if hasattr(module, 'weight'):
            # Accessing the property directly returns the parametrized (quantized) weight
            w = module.weight

            # Determine if the weight is a trainable parameter by checking 'original' if parametrized
            is_trainable = False
            if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                is_trainable = module.parametrizations.weight.original.requires_grad
            elif isinstance(w, torch.nn.Parameter):
                is_trainable = w.requires_grad

            if is_trainable and w.dim() >= 2:
                # We use w.data to ensure we are working with the tensor values
                planes = get_bitplanes(w.data, bit_depth, robust=robust, percentile=percentile)
    def chunk_list(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        m_chunks = list(chunk_list(planes, max(1, bit_depth // (max_workers * 2))))
        m_results = list(executor.map(bdm_batch_worker, m_chunks))
        total_multi = sum(sum(res) for res in m_results)

    return total_multi, np.array(m_results).flatten()


def multi_plane_ratio(c_multi, s_multi):
    """Total QuBD-style complexity ratio across all planes, as a percentage: matches the
    paper's Eq. 14 definition (Delta C_QuBD = C_QuBD(f_PT) / C_QuBD(f_RND)), where
    C_QuBD(w) = sum of per-plane complexity (Eq. 2/Fig. 2's "Aggregate" step) -- i.e. a
    plain ratio(sum(c_multi), sum(s_multi)), not a per-plane average or weighting.

    c_multi/s_multi are per-plane score lists from measure_complexity()'s total_multi[bd].
    Note this aggregate dilutes as bit_depth grows and more planes are near-random (see the
    paper's Fig. 7 discussion): to see *where* the reduction is concentrated (MSB vs LSB),
    use the per-plane ratios directly (c_multi[i]/s_multi[i]) rather than this aggregate --
    that's what Fig. 1/6/7 plot, and what this function intentionally does not compute.
    """
    total_s = sum(s_multi)
    if not total_s:
        return 0.0
    return 100.0 * sum(c_multi) / total_s


def per_plane_ratio(c_multi, s_multi):
    """Per-plane complexity ratio c_multi[i]/s_multi[i], as percentages (list ordered LSB
    index 0 -> MSB index bd-1). This is the quantity to plot against plane index to show
    where structural reduction is concentrated (paper Fig. 1/6/7), since the aggregate
    multi_plane_ratio() dilutes that signal once several planes are near-random.
    """
    return [100.0 * c / s if s else 0.0 for c, s in zip(c_multi, s_multi)]


def measure_complexity(model, bit_depths=[8], max_workers=8, robust=False, percentile=99.9, tiled=False):
    """Aggregates complexity scores for independent and optionally tiled manifolds."""
    binary_tasks = []
    multi_tasks = {bd: [] for bd in bit_depths}
    tiled_tasks = {bd: [] for bd in bit_depths} if tiled else {}
    tensors = []
    # Identify tensors to process: either from a model or a direct tensor input
    if isinstance(model, nn.Module):
        for name, module in model.named_modules():
            if hasattr(module, 'weight'):
                # Accessing the property directly returns the parametrized (quantized) weight
                w = module.weight

                # Determine if the weight is a trainable parameter by checking 'original' if parametrized
                is_trainable = False
                if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
                    is_trainable = module.parametrizations.weight.original.requires_grad
                elif isinstance(w, torch.nn.Parameter):
                    is_trainable = w.requires_grad
                if is_trainable and w.dim() >= 2:
                    tensors.append(w)
    elif torch.is_tensor(model):
    # Input is a raw tensor: proceed directly if dimensions match
        if model.dim() >= 2:
            tensors.append(model)

    for p_data in tensors:
        p_val = p_data.data
        # We use w.data to ensure we are working with the tensor values
        binary_tasks.append( (p_val > 0).cpu().numpy().astype(np.int8))
        for bd in bit_depths:
            if tiled:
                tm = get_tiled_manifold(p_val, bd, robust=robust, percentile=percentile)
                if tm is not None:
                    tiled_tasks[bd].append(tm)

            planes = get_bitplanes(p_val, bd, robust=robust, percentile=percentile)
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
            
            flat = [score for chunk in m_results for score in chunk]
            total_multi[bd] = [sum(flat[i::bd]) for i in range(bd)] # re-group by plane

            if tiled:
                t_tasks = tiled_tasks[bd]
                t_chunks = list(chunk_list(t_tasks, max(1, len(t_tasks) // max_workers)))
                t_results = list(executor.map(bdm_batch_worker, t_chunks))
                total_tiled[bd] = sum(sum(res) for res in t_results)

        return float(total_bin), total_multi, total_tiled



