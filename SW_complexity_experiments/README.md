# q-Bit Quantized BDM

## Scripts

### complexity_per_layer.py
Computes bitplane complexity per layer for 5 pretrained models (ResNet18, ResNet50, ViT-B/16, EfficientNet-B0, MobileNetV3), comparing pretrained vs. random weights. Results are saved to `results/complexity_per_layer_5models.json`.

```
python complexity_per_layer.py
```

### complexity_per_model.py
Computes whole-model complexity for a list of timm models, comparing pretrained vs. random weights. Model names are read from `model_names_100.txt`. Results are saved to `results/complexities_100models.json`.

```
python complexity_per_model.py
```
