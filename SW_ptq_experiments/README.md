# q-Bit Quantized BDM

## Scripts

### ptq.py
Evaluates post-training quantization (PTQ) accuracy for 5 pretrained models (ResNet18, ResNet50, ViT-B/16, EfficientNet-B0, MobileNetV3) on the ImageNet-1K validation set. Compares FP32, FP16, and per-channel uniform PTQ at bit depths 1–8. Results are saved to `ptq.json`.

```
python ptq.py
```

### ptq_utils.py
Utility module providing quantizer classes (`UniformQuantizer`, `UniformQuantizer_per_channel`) and helper functions (`attach_weight_quantizers`, `detach_weight_quantizers`, `toggle_quantization`) used by `ptq.py`. Not intended to be run directly.
