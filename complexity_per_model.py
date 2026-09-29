import os
os.environ["PYTHONWARNINGS"] = "ignore"

import json
import timm
from qbdm.qbdm import measure_complexity
from utils.random_init import get_fixed_random_model

MODEL_NAMES_FILE = "model_names_100.txt" # "model_names_5.txt"

with open(MODEL_NAMES_FILE, "r") as f:
    MODELS = [line.strip() for line in f if line.strip()]

BIT_DEPTHS = [8] #[1, 2, 4, 8, 16, 32]

# Robust Normalization Parameters (percentile-clipped quantizer range, as in train.py)
USE_ROBUST_NORM = True
ROBUST_PERCENTILE = 99.9

# Output name reflects the settings, e.g. complexities_100models_robust_p99.9.json or
# complexities_5models_bd16_robust_p99.9.json; bit depths are only tagged when not [8].
BD_TAG = "" if BIT_DEPTHS == [8] else "_bd" + "-".join(str(bd) for bd in BIT_DEPTHS)
SUFFIX = f"_robust_p{ROBUST_PERCENTILE:g}" if USE_ROBUST_NORM else ""
OUT_PATH = f"results/complexities_{len(MODELS)}models{BD_TAG}{SUFFIX}.json"

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

    # Check that all models are available in timm. If not, print warning and exit.
    # Names may include a pretrained tag (e.g. vit_base_patch16_224.augreg_in1k), which
    # list_models() does not contain, so also check list_pretrained()
    available_models = set(timm.list_models()) | set(timm.list_pretrained())
    missing = [m for m in MODELS if m not in available_models]
    if missing:
        print(f"Error: the following models were not found in timm:")
        for m in missing:
            print(f"  - {m}")
        exit(1)

    for model_name in MODELS:
        if model_name in results:
            continue
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

        # Save after every model so an interrupted run keeps what it has computed so far
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print(f"\nSaved to {OUT_PATH}")