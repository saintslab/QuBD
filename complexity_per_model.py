import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import timm
from qbdm.qbdm import measure_complexity
from utils.random_init import get_fixed_random_model

with open("model_names_100.txt", "r") as f:
    MODELS = [line.strip() for line in f if line.strip()]

BIT_DEPTHS = [8] #[1, 2, 4, 8, 16, 32]

# Robust Normalization Parameters (percentile-clipped quantizer range, as in train.py)
USE_ROBUST_NORM = True
ROBUST_PERCENTILE = 99.9

SUFFIX = f"_robust_p{ROBUST_PERCENTILE:g}" if USE_ROBUST_NORM else ""
OUT_PATH = f"results/complexities_100models{SUFFIX}.json"

def count_params(model):
    return sum(p.numel() for p in model.parameters())

if __name__ == '__main__':
    results = {
        "bit_depths": BIT_DEPTHS,
        "models": MODELS
    }

    # Check that all models are available in timm. If not, print warning and exit.
    available_models = timm.list_models()
    missing = [m for m in MODELS if m not in available_models]
    if missing:
        print(f"Error: the following models were not found in timm:")
        for m in missing:
            print(f"  - {m}")
        exit(1)

    for model_name in MODELS:
        print(f"\nProcessing {model_name}...")
        try:
            model_pre = timm.create_model(model_name, pretrained=True).eval()
            model_ran = get_fixed_random_model(model_name)

            bin_p, qbit_p, _ = measure_complexity(model_pre, bit_depths=BIT_DEPTHS, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
            bin_r, qbit_r, _ = measure_complexity(model_ran, bit_depths=BIT_DEPTHS, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)

            results[model_name] = {
                "num_params": count_params(model_pre),
                "binary": {"pretrained": bin_p, "random": bin_r},
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

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {OUT_PATH}")