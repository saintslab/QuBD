import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import numpy as np
import timm
import torch.nn as nn
from concurrent.futures import ProcessPoolExecutor
from qbdm.qbdm import get_bitplanes, bdm_batch_worker
from utils.random_init import get_fixed_random_model

# Ablation: is the QuBD complexity ratio (pretrained / random) robust to how 4D conv weights
# [C_out, C_in, H, W] are reshaped to 2D? get_bitplanes uses w.flatten(1) -> [C_out, C_in*H*W].
# The same weights are reshaped in three other ways and the per-layer, per-plane ratios are
# compared with the default. Quantization depends only on the weight values, so every reshape
# gets identical quantized values; only their arrangement (and hence BDM's 4x4 blocks) changes.

MODELS = ["resnet18", "resnet50"]
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

RESHAPES = {
    "default": lambda w: w.flatten(1),                                       # [C_out, C_in*H*W]
    "in_channel_rows": lambda w: w.permute(1, 0, 2, 3).contiguous().flatten(1),  # [C_in, C_out*H*W]
    "channels_last": lambda w: w.permute(0, 2, 3, 1).contiguous().flatten(1),    # [C_out, H*W*C_in]
    "square": square,
}

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def layer_ratios(model_pre, model_ran):
    """Per-layer, per-plane ratio (%) for every conv layer and reshape. A reshape is skipped for a
    layer if it gives the same matrix as the default (e.g. channels_last on 1x1 convs) or a matrix
    with fewer than 4 rows or columns (which get_bitplanes cannot score)."""
    random_modules = dict(model_ran.named_modules())
    tasks, keys = [], []
    for name, module in model_pre.named_modules():
        if not (isinstance(getattr(module, "weight", None), nn.Parameter) and module.weight.dim() == 4):
            continue
        w_pre, w_ran = module.weight.data, random_modules[name].weight.data
        default_pre = RESHAPES["default"](w_pre)
        for variant, reshape in RESHAPES.items():
            m_pre, m_ran = reshape(w_pre), reshape(w_ran)
            if variant != "default" and m_pre.shape == default_pre.shape and bool((m_pre == default_pre).all()):
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

def compare(ratios):
    """For each reshape vs. the default, per bit plane over the layers both have: Pearson r across
    layers, mean and max absolute difference (percentage points), and the layer with the max."""
    stats = {}
    for variant, layers in ratios.items():
        if variant == "default":
            continue
        common = [name for name in ratios["default"] if name in layers]
        base = np.array([ratios["default"][name] for name in common])
        other = np.array([layers[name] for name in common])
        diff = np.abs(other - base)
        stats[variant] = {"layers": common, "per_plane": {
            str(p): {"pearson": float(np.corrcoef(base[:, p], other[:, p])[0, 1]),
                     "mean_abs_diff": float(diff[:, p].mean()),
                     "max_abs_diff": float(diff[:, p].max()),
                     "max_layer": common[int(diff[:, p].argmax())]}
            for p in range(BIT_DEPTH)}}
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
        print(f"  {'vs. default':16s} {'layers':>7s}   Pearson r (plane 7..0)                    max |diff| in pp (plane 7..0)")
        for variant, s in stats.items():
            pp = [s["per_plane"][str(p)] for p in range(BIT_DEPTH - 1, -1, -1)]
            print(f"  {variant:16s} {len(s['layers']):>3d}/{n_conv:<3d}   " + " ".join(f"{x['pearson']:4.2f}" for x in pp)
                  + "   " + " ".join(f"{x['max_abs_diff']:4.1f}" for x in pp))

    with open(OUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
