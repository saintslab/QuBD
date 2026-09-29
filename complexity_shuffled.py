import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import torch
import torch.nn as nn
import timm
from qbdm.qbdm import measure_complexity
from utils.random_init import get_fixed_random_model

# Shuffled control for complexity_per_model.py: the same QuBD complexities, but with the values of
# every scored weight tensor (trainable module weights with dim >= 2) randomly permuted in both the
# pretrained and the random model. This keeps each tensor's value distribution but removes all
# ordering, so comparing with complexity_per_model.py's results shows how much of the
# pretrained/random complexity ratio comes from the arrangement of the weights.

MODEL_NAMES_FILE = "model_names_100.txt" # "model_names_5.txt"

with open(MODEL_NAMES_FILE, "r") as f:
    MODELS = [line.strip() for line in f if line.strip()]

BIT_DEPTHS = [8]

# Robust Normalization Parameters (percentile-clipped quantizer range, as in train.py)
USE_ROBUST_NORM = True
ROBUST_PERCENTILE = 99.9

BD_TAG = "" if BIT_DEPTHS == [8] else "_bd" + "-".join(str(bd) for bd in BIT_DEPTHS)
SUFFIX = f"_robust_p{ROBUST_PERCENTILE:g}" if USE_ROBUST_NORM else ""
OUT_PATH = f"results/complexities_{len(MODELS)}models{BD_TAG}_shuffled{SUFFIX}.json"

def shuffle_weights(model):
    """Randomly permutes the values of every scored weight tensor, with the same fixed permutation
    for every tensor of a given size (so pretrained and random weights are shuffled identically)."""
    with torch.no_grad():
        for _, module in model.named_modules():
            w = getattr(module, "weight", None)
            if isinstance(w, nn.Parameter) and w.requires_grad and w.dim() >= 2:
                perm = torch.randperm(w.numel(), generator=torch.Generator().manual_seed(0))
                w.copy_(w.reshape(-1)[perm].reshape(w.shape))
    return model

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    results = {
        "bit_depths": BIT_DEPTHS,
        "models": MODELS
    }

    # Resume a previous (e.g. crashed) run with the same settings: models already in
    # OUT_PATH are skipped. Delete OUT_PATH to start from scratch.
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "r") as f:
            previous = json.load(f)
        if previous["bit_depths"] != BIT_DEPTHS or previous["models"] != MODELS:
            print(f"Error: {OUT_PATH} was made with different models or bit depths; move or delete it first.")
            exit(1)
        results = previous
        done = [m for m in MODELS if m in results]
        print(f"Resuming from {OUT_PATH}: {len(done)}/{len(MODELS)} models already done.")

    for model_name in MODELS:
        if model_name in results:
            continue
        print(f"\nProcessing {model_name}...")
        try:
            model_pre = shuffle_weights(timm.create_model(model_name, pretrained=True).eval())
            model_ran = shuffle_weights(get_fixed_random_model(model_name))

            _, qbit_p, _ = measure_complexity(model_pre, bit_depths=BIT_DEPTHS, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
            _, qbit_r, _ = measure_complexity(model_ran, bit_depths=BIT_DEPTHS, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)

            results[model_name] = {
                "num_params": count_params(model_pre),
                "bitplane": {
                    bd: {
                        "pretrained": qbit_p[bd],
                        "random": qbit_r[bd]
                    } for bd in BIT_DEPTHS
                }
            }
            print(f"  Done.")

        except Exception as e:
            print(f"  Failed: {e}")

        # Save after every model so an interrupted run keeps what it has computed so far
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {OUT_PATH}")
