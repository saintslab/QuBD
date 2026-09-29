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

# Ablation: is the model-level QuBD complexity ratio (pretrained / random) robust to how 4D conv
# weights [C_out, C_in, H, W] are reshaped to 2D? get_bitplanes uses w.flatten(1) ->
# [C_out, C_in*H*W]. Here the conv weights are reshaped in four other ways; 2D (linear) weights
# are unaffected. Quantization depends only on the weight values, so every reshape gets identical
# quantized values; only their arrangement (and hence BDM's 4x4 blocks) changes. As a control,
# "shuffled" randomly permutes the values of every weight tensor (conv and linear), which keeps
# the value distribution but removes all ordering.
# Complexities are summed over all weight matrices scored by measure_complexity, per bit plane,
# for pretrained and random weights; the ratio of the sums is the model-level QuBD ratio.

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

# Reshapes of 4D conv weights; "shuffled" applies to every weight tensor.
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

def scored_weights(model):
    """The weight tensors measure_complexity scores: trainable module weights with dim >= 2."""
    out = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight"):
            continue
        w = module.weight
        if hasattr(module, "parametrizations") and "weight" in module.parametrizations:
            trainable = module.parametrizations.weight.original.requires_grad
        else:
            trainable = isinstance(w, nn.Parameter) and w.requires_grad
        if trainable and w.dim() >= 2:
            out.append((name, w.data))
    return out

def valid(m):
    return m.shape[0] >= 4 and m.shape[1] >= 4

def score(pairs):
    """BDM complexity per bit plane for a list of (pretrained matrix, random matrix)."""
    tasks, keys = [], []
    for j, (m_pre, m_ran) in enumerate(pairs):
        for kind, m in (("pretrained", m_pre), ("random", m_ran)):
            planes = get_bitplanes(m, BIT_DEPTH, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
            tasks.extend(planes)
            keys.extend((j, kind, i) for i in range(len(planes)))
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        chunks = list(chunk_list(tasks, max(1, len(tasks) // (MAX_WORKERS * 2))))
        scores = [s for chunk in executor.map(bdm_batch_worker, chunks) for s in chunk]
    out = [{"pretrained": [0.0] * BIT_DEPTH, "random": [0.0] * BIT_DEPTH} for _ in pairs]
    for (j, kind, i), s in zip(keys, scores):
        out[j][kind][i] = s
    return out

def model_ablation(model_pre, model_ran):
    """Per-layer complexities for every reshape, and the model-level ratio per bit plane. A layer
    keeps its default complexities under a reshape that does not apply to it (2D weights for the
    conv reshapes), gives the default matrix or its transpose (which BDM scores identically), or
    gives fewer than 4 rows or columns; so every reshape scores the same set of weights."""
    random_weights = dict(scored_weights(model_ran))
    layers = [(name, w, random_weights[name]) for name, w in scored_weights(model_pre)]
    total = sum(w.numel() for _, w, _ in layers if valid(RESHAPES["default"](w)))

    default = score([(RESHAPES["default"](wp), RESHAPES["default"](wr)) for _, wp, wr in layers])
    per_layer = {"default": {name: c for (name, _, _), c in zip(layers, default)}}
    results = {}
    for variant, reshape in RESHAPES.items():
        changed = []
        for name, wp, wr in layers:
            d = RESHAPES["default"](wp)
            if not valid(d) or (wp.dim() != 4 and variant != "shuffled"):
                continue
            m = reshape(wp)
            if not valid(m) or any(m.shape == x.shape and bool((m == x).all()) for x in (d, d.T)):
                continue
            changed.append((name, m, reshape(wr)))
        if variant != "default":
            per_layer[variant] = {name: c for (name, _, _), c in zip(changed, score([(mp, mr) for _, mp, mr in changed]))}
        layer_c = {**per_layer["default"], **per_layer[variant]}
        pre = np.sum([layer_c[name]["pretrained"] for name, _, _ in layers], axis=0)
        ran = np.sum([layer_c[name]["random"] for name, _, _ in layers], axis=0)
        results[variant] = {"ratio": (pre / ran * 100).tolist(),
                            "weights_rearranged": 0.0 if variant == "default" else sum(mp.numel() for _, mp, _ in changed) / total}
    return results, per_layer

if __name__ == '__main__':
    all_results = {}
    for model_name in MODELS:
        print(f"\nProcessing {model_name}...")
        model_pre = timm.create_model(model_name, pretrained=True).eval()
        model_ran = get_fixed_random_model(model_name)
        results, per_layer = model_ablation(model_pre, model_ran)
        all_results[model_name] = {"model_level": results, "per_layer_complexity": per_layer}

        base = results["default"]["ratio"]
        print(f"  {'':16s} {'weights':>8s}   model-level ratio, plane 7..0 (for other reshapes: difference to default, pp)")
        print(f"  {'default':16s} {'':>8s}   " + " ".join(f"{base[p]:6.1f}" for p in range(BIT_DEPTH - 1, -1, -1)))
        for variant, r in results.items():
            if variant == "default":
                continue
            print(f"  {variant:16s} {r['weights_rearranged'] * 100:7.1f}%   "
                  + " ".join(f"{r['ratio'][p] - base[p]:+6.1f}" for p in range(BIT_DEPTH - 1, -1, -1)))

    with open(OUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
