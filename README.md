# QuBD Complexity 

Official repository for [Bakhtiarifard et al. (2026)](https://arxiv.org/abs/2605.15551) "Characterizing Learning in Deep Neural Networks using Tractable Algorithmic Complexity Analysis", published at NeurIPS 2026. 

![qubd](utils/qubd.png)

## Scripts

### Main Results
Results in Figure 4 in the paper can be reproduced by running this script:

```
python train.py
```

The scripts below use one of two model lists: `model_names_100.txt` (100 timm models) or `model_names_5.txt` (ResNet18, ResNet50, ViT-B/16, EfficientNet-B0, MobileNetV3). Settings such as the model list, bit depths and clipping are set at the top of each script; by default the weights are clipped at the 99.9th percentile, as in `train.py`. All results are saved to `results/`, with the settings in the filename.

### Complexity across models
Computes whole-model complexity for the models in `model_names_100.txt`, comparing pretrained vs. random weights. Results are saved after every model, so an interrupted run resumes where it stopped.

```
python complexity_per_model.py
```

### Complexity per layer 
Computes bit-plane complexity per layer for the models in `model_names_5.txt`, comparing pretrained vs. random weights.

```
python complexity_per_layer.py
```

###  Post-Training Quantization and QuBD Complexity

Evaluates post-training quantization (PTQ) accuracy for the models in `model_names_5.txt` on the ImageNet-1K validation set. Compares FP32, FP16, and per-channel uniform PTQ at bit depths 1–8.

```
python ptq.py
```

Utility module providing quantizer classes (`UniformQuantizer`, `UniformQuantizer_per_channel`) and helper functions (`attach_weight_quantizers`, `detach_weight_quantizers`, `toggle_quantization`) used by `ptq.py`. Not intended to be run directly.

### Plotting
`plot_figures.py` makes the figures from the results above and saves them to `figs/`, with the settings in the filename. Choose the clipping with `--percentile` (e.g. `99.99`, or `none`; default 99.9).

| Figure | Uses results from |
|---|---|
| `ranked100`: complexity per bit-plane for all 100 models | `complexity_per_model.py` |
| `per_layer`: complexity per layer for one model | `complexity_per_layer.py` |
| `complexity_accuracy`: complexity per bit-plane next to PTQ accuracy | `complexity_per_model.py` (with `model_names_5.txt`, 16-bit) and `ptq.py` |

```
python plot_figures.py ranked100 --percentile 99.9 --panels 4
python plot_figures.py per_layer --model resnet50 --percentile none
python plot_figures.py complexity_accuracy --percentile 99.99
python plot_figures.py all    # every figure for every clipping setting (none, 99.99, 99.9); skips missing results
```

### Usage guidelines ###

* Kindly cite our publication if you use any part of the code

```
@inproceedings{bakh2026qubd,
        title={{Characterizing Learning in Deep Neural Networks using Tractable Algorithmic Complexity Analysis}},
        author={Pedram Bakhtiarifard, Sophia N. Wilson, Mahmoud Afifi, Jonathan Wenshøj and Raghavendra Selvan},
        booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
        note={arXiv preprint arxiv:2605.15551},
        year={2026}}
```
### Who do I talk to? ###

* pba@di.ku.dk; raghav@di.ku.dk

