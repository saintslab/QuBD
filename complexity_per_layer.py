import os
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import timm
import torch.nn as nn
import json
from concurrent.futures import ProcessPoolExecutor
from qbdm.qbdm import get_bitplanes, bdm_batch_worker
from utils.random_init import get_fixed_random_model

with open("model_names_5.txt", "r") as f:
    MODELS = [line.strip() for line in f if line.strip()]

BIT_DEPTH = 8
MAX_WORKERS = 8

# Robust Normalization Parameters (percentile-clipped quantizer range, as in train.py)
USE_ROBUST_NORM = True
ROBUST_PERCENTILE = 99.9

SUFFIX = f"_robust_p{ROBUST_PERCENTILE:g}" if USE_ROBUST_NORM else ""
OUT_PATH = f"results/complexity_per_layer_5models{SUFFIX}.json"

def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':
    all_results = {}

    for model_name in MODELS:
        print(f"\nProcessing {model_name}...")
        model_pre = timm.create_model(model_name, pretrained=True).eval()
        model_ran = get_fixed_random_model(model_name)
        random_modules = dict(model_ran.named_modules())

        # Collect the bit-planes of every layer (pretrained and random) first and score them all
        # in one shared worker pool, instead of starting a new pool for every layer.
        tasks, keys = [], []
        for name, module in model_pre.named_modules():
            if not (hasattr(module, "weight") and isinstance(module.weight, nn.Parameter)):
                continue
            if module.weight.dim() < 2:
                continue

            for kind, w in (("pretrained", module.weight.data), ("random", random_modules[name].weight.data)):
                planes = get_bitplanes(w, BIT_DEPTH, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
                tasks.extend(planes)
                keys.extend((name, kind, i) for i in range(len(planes)))

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            chunks = list(chunk_list(tasks, max(1, len(tasks) // (MAX_WORKERS * 2))))
            scores = [score for chunk in executor.map(bdm_batch_worker, chunks) for score in chunk]

        qbit = {}
        for (name, kind, i), score in zip(keys, scores):
            qbit.setdefault(name, {"pretrained": [0.0] * BIT_DEPTH, "random": [0.0] * BIT_DEPTH})[kind][i] = score

        all_results[model_name] = {
            name: [q["pretrained"][i] / q["random"][i] * 100 for i in range(BIT_DEPTH)]
            for name, q in qbit.items()
        }

    with open(OUT_PATH, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
