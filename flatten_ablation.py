import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import numpy as np
import timm
import torch
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
from qbdm.qbdm import get_bitplanes, bdm_batch_worker
from utils.random_init import get_fixed_random_model

# Ablation: is the QuBD complexity ratio (pretrained / random) robust to how 4D conv weights
# [C_out, C_in, H, W] are reshaped to 2D? get_bitplanes uses w.flatten(1) -> [C_out, C_in*H*W].
# The same weights are reshaped in four other ways and the per-layer, per-plane ratios are
# compared with the default. Quantization depends only on the weight values, so every reshape
# gets identical quantized values; only their arrangement (and hence BDM's 4x4 blocks) changes.
# As a control, "shuffled" randomly permutes each tensor's values (fixed seed), which keeps the
# value distribution but removes all ordering.

with open("model_names_5.txt", "r") as f:
    MODELS = [line.strip() for line in f if line.strip()]

BIT_DEPTH = 8
MAX_WORKERS = 8

# Robust Normalization Parameters (percentile-clipped quantizer range, as in train.py)
USE_ROBUST_NORM = True
ROBUST_PERCENTILE = 99.9

SUFFIX = f"_robust_p{ROBUST_PERCENTILE:g}" if USE_ROBUST_NORM else ""
OUT_PATH = f"results/flatten_ablation{SUFFIX}.json"

def square(w):
    """Shape-agnostic: the raw flat buffer as the closest-to-square (r, c) with r * c = numel."""
    n = w.numel()
    r = int(np.floor(np.sqrt(n)))
    while r > 1 and n % r != 0:
        r -= 1
    return w.reshape(r, n // r)

def mosaic(w):
    """Each kernel kept as its 2D H x W patch: [C_out*H, C_in*W], a grid of kernels. Equals the
    default for 1x1 convs (and so for linear layers)."""
    c_out, c_in, h, k = w.shape
    return w.permute(0, 2, 1, 3).contiguous().reshape(c_out * h, c_in * k)

def shuffled(w):
    """Control: the tensor's values in a random order (the same permutation for every tensor of
    this size, so pretrained and random weights are shuffled identically), in the default shape."""
    perm = torch.randperm(w.numel(), generator=torch.Generator().manual_seed(0))
    return w.reshape(-1)[perm].reshape(w.shape).flatten(1)

RESHAPES = {
    "default": lambda w: w.flatten(1),                                       # [C_out, C_in*H*W]
    "in_channel_rows": lambda w: w.permute(1, 0, 2, 3).contiguous().flatten(1),  # [C_in, C_out*H*W]
    "channels_last": lambda w: w.permute(0, 2, 3, 1).contiguous().flatten(1),    # [C_out, H*W*C_in]
    "square": square,
    "mosaic": mosaic,                                                        # [C_out*H, C_in*W]
    "shuffled": shuffled,
}

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def layer_ratios(model_pre, model_ran):
    """Per-layer, per-plane ratio (%) for every conv layer and reshape. A reshape is skipped for a
    layer if it gives the same matrix as the default or its transpose, which BDM scores identically
    (e.g. channels_last and in_channel_rows on 1x1 convs), or a matrix with fewer than 4 rows or
    columns (which get_bitplanes cannot score)."""
    random_modules = dict(model_ran.named_modules())
    tasks, keys = [], []
    for name, module in model_pre.named_modules():
        if not (isinstance(getattr(module, "weight", None), nn.Parameter) and module.weight.dim() == 4):
            continue
        w_pre, w_ran = module.weight.data, random_modules[name].weight.data
        default_pre = RESHAPES["default"](w_pre)
        for variant, reshape in RESHAPES.items():
            m_pre, m_ran = reshape(w_pre), reshape(w_ran)
            if variant != "default" and any(m_pre.shape == d.shape and bool((m_pre == d).all())
                                            for d in (default_pre, default_pre.T)):
                continue
            for kind, m in (("pretrained", m_pre), ("random", m_ran)):
                planes = get_bitplanes(m, BIT_DEPTH, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
                tasks.extend(planes)
                keys.extend((variant, name, kind, i) for i in range(len(planes)))

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunks = list(chunk_list(tasks, max(1, len(tasks) // (MAX_WORKERS * 2))))
        scores = [score for chunk in executor.map(bdm_batch_worker, chunks) for score in chunk]

    qbit = {}
    for (variant, name, kind, i), score in zip(keys, scores):
        qbit.setdefault(variant, {}).setdefault(name, {"pretrained": [0.0] * BIT_DEPTH, "random": [0.0] * BIT_DEPTH})[kind][i] = score
    return {variant: {name: [q["pretrained"][i] / q["random"][i] * 100 for i in range(BIT_DEPTH)]
                      for name, q in layers.items()}
            for variant, layers in qbit.items()}

def pearson(a, b):
    """Pearson r, or None if there are too few layers or no spread to correlate."""
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])

def compare(ratios):
    """For each reshape vs. the default, per bit plane over the layers both have: mean ratio of the
    default and of the reshape, Pearson r across layers, mean and max absolute difference
    (percentage points), and the layer with the max difference."""
    stats = {}
    for variant in RESHAPES:
        if variant == "default":
            continue
        common = [name for name in ratios["default"] if name in ratios.get(variant, {})]
        stats[variant] = {"layers": common, "per_plane": {}}
        if not common:
            continue
        base = np.array([ratios["default"][name] for name in common])
        other = np.array([ratios[variant][name] for name in common])
        diff = np.abs(other - base)
        stats[variant]["per_plane"] = {
            str(p): {"mean_ratio_default": float(base[:, p].mean()),
                     "mean_ratio": float(other[:, p].mean()),
                     "pearson": pearson(base[:, p], other[:, p]),
                     "mean_abs_diff": float(diff[:, p].mean()),
                     "max_abs_diff": float(diff[:, p].max()),
                     "max_layer": common[int(diff[:, p].argmax())]}
            for p in range(BIT_DEPTH)}
    return stats

if __name__ == '__main__':
    all_results = {}
    for model_name in MODELS:
        print(f"\nProcessing {model_name}...")
        model_pre = timm.create_model(model_name, pretrained=True).eval()
        model_ran = get_fixed_random_model(model_name)
        ratios = layer_ratios(model_pre, model_ran)
        stats = compare(ratios)
        all_results[model_name] = {"per_layer_ratios": ratios, "comparison": stats}

        n_conv = len(ratios["default"])
        print(f"  {'vs. default':16s} {'layers':>7s}  {'mean ratio P7':>15s}   Pearson r (plane 7..0)"
              f"                    max |diff| in pp (plane 7..0)")
        for variant, s in stats.items():
            if not s["layers"]:
                print(f"  {variant:16s} {0:>3d}/{n_conv:<3d}  (no layers)")
                continue
            pp = [s["per_plane"][str(p)] for p in range(BIT_DEPTH - 1, -1, -1)]
            print(f"  {variant:16s} {len(s['layers']):>3d}/{n_conv:<3d}  {pp[0]['mean_ratio_default']:5.1f} -> {pp[0]['mean_ratio']:5.1f}   "
                  + " ".join("  - " if x["pearson"] is None else f"{x['pearson']:4.2f}" for x in pp)
                  + "   " + " ".join(f"{x['max_abs_diff']:4.1f}" for x in pp))

    with open(OUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
