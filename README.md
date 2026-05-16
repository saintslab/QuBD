# q-Bit Quantized BDM

Try it out by
```
python test.py
```
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
# q-Bit Quantized BDM

## Scripts

### ptq.py
Evaluates post-training quantization (PTQ) accuracy for 5 pretrained models (ResNet18, ResNet50, ViT-B/16, EfficientNet-B0, MobileNetV3) on the ImageNet-1K validation set. Compares FP32, FP16, and per-channel uniform PTQ at bit depths 1–8. Results are saved to `ptq.json`.

```
python ptq.py
```

### ptq_utils.py
Utility module providing quantizer classes (`UniformQuantizer`, `UniformQuantizer_per_channel`) and helper functions (`attach_weight_quantizers`, `detach_weight_quantizers`, `toggle_quantization`) used by `ptq.py`. Not intended to be run directly.
