import os
os.environ["PYTHONWARNINGS"] = "ignore"

import torch
import timm
import torch.nn as nn
import json
from qbdm.qbdm import measure_complexity

MODELS = [
    "resnet18",
    "resnet50",
    "vit_base_patch16_224",
    "efficientnet_b0",
    "mobilenetv3_large_100",
]
BIT_DEPTH = 8

if __name__ == '__main__':
    all_results = {}

    for model_name in MODELS:
        torch.manual_seed(42)  # Reset seed for reproducibility
        print(f"\nProcessing {model_name}...")
        model_pre = timm.create_model(model_name, pretrained=True).eval()
        model_ran = timm.create_model(model_name, pretrained=False).eval()
        random_modules = dict(model_ran.named_modules())
        results = {}

        for name, module in model_pre.named_modules():
            if not (hasattr(module, "weight") and isinstance(module.weight, nn.Parameter)):
                continue
            if module.weight.dim() < 2:
                continue

            w_pre = module.weight.data
            w_ran = random_modules[name].weight.data

            _, qbit_p, _ = measure_complexity(w_pre, bit_depths=[BIT_DEPTH])
            _, qbit_r, _ = measure_complexity(w_ran, bit_depths=[BIT_DEPTH])

            plane_ratios = [
                qbit_p[BIT_DEPTH][i] / qbit_r[BIT_DEPTH][i] * 100
                for i in range(BIT_DEPTH)
            ]

            results[name] = plane_ratios

        all_results[model_name] = results

    with open('results/complexity_per_layer_5models.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved to results/complexity_per_layer_5models.json")