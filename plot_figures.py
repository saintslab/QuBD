"""Figures for the QuBD complexity analysis. Run from the repository root, e.g.

    python plot_figures.py per_plane --percentile 99.9 --planes 4
    python plot_figures.py per_layer --model resnet18 --percentile none
    python plot_figures.py vs_accuracy --percentile 99.99
    python plot_figures.py all

--percentile selects the clipping setting of the results to plot ("none" = no clipping)
and must match a run of complexity_per_model.py / complexity_per_layer.py. Figures are
saved to figs/ with the model, clipping and bit depth in the filename.
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from utils.pedram_style import (use_pedram_style, apply_panel_style, PALETTE,
                                two_panel_size, four_panel_row_wide_size, two_row_four_panel_wide_size)

NAMES = {"resnet18": "ResNet-18", "resnet50": "ResNet-50", "vit_base_patch16_224": "ViT-B/16",
         "efficientnet_b0": "EfficientNet-B0", "mobilenetv3_large_100": "MobileNetV3"}
DELTA_C = r"$\Delta C_{\mathrm{QuBD}}$ (\%)"


def suffix(p):
    """Results-file suffix used by the complexity scripts for clipping percentile p."""
    return "" if p is None else f"_robust_p{p:g}"


def tag(p):
    return "noclip" if p is None else f"clip{p:g}"


def load(path):
    with open(path) as f:
        return json.load(f)


def ratios(data, model, bit_depth):
    """Pretrained/random complexity ratio (%) per bit plane (index 0 = LSB)."""
    bp = data[model]["bitplane"][str(bit_depth)]
    return np.array(bp["pretrained"]) / np.array(bp["random"]) * 100


def save(fig, name):
    os.makedirs("figs", exist_ok=True)
    fig.savefig(f"figs/{name}.pdf", bbox_inches="tight", dpi=1200)
    plt.close(fig)
    print(f"Saved figs/{name}.pdf")


def per_plane(p, planes=4):
    """Per-plane ratio of all 100 models, ranked by the non-clipped MSB ratio, marker size ~ #params."""
    data = load(f"results/complexities_100models{suffix(p)}.json")
    models = [m for m in data["models"] if m in data]
    # Rank by the non-clipped MSB ratio so every clipping setting shares the same x-axis order
    rank_data = load("results/complexities_100models.json") if os.path.exists("results/complexities_100models.json") else data
    models.sort(key=lambda m: -ratios(rank_data, m, 8)[7] if m in rank_data else 0)
    max_params = max(data[m]["num_params"] for m in models)
    size = lambda n_params: n_params / max_params * 100 + 20

    shown = [7, 5, 3, 1] if planes == 4 else list(range(7, -1, -1))
    figsize = four_panel_row_wide_size() if planes == 4 else two_row_four_panel_wide_size()
    fig, axes = plt.subplots(1 if planes == 4 else 2, 4, figsize=figsize, sharey=True)
    axes = axes.flatten()
    for ax, plane in zip(axes, shown):
        r = [ratios(data, m, 8)[plane] for m in models]
        ax.axhline(100, color="gray", linestyle="--", linewidth=2, alpha=0.6, zorder=2)
        sc = ax.scatter(range(len(models)), r, s=[size(data[m]["num_params"]) for m in models], c=r,
                        cmap="RdYlGn_r", vmin=0, vmax=100, edgecolors="black", linewidths=0.5, zorder=4)
        ax.set_title(f"Plane {plane}" + {7: " (MSB)", 0: " (LSB)"}.get(plane, ""))
        ax.set_xticks(np.arange(0, len(models) + 1, 10))
        apply_panel_style(ax)
    for ax in axes[-4:]:
        ax.set_xlabel(r"Models Ranked by $\Delta C_{\mathrm{QuBD}}$ (MSB)")
    for ax in axes[::4]:
        ax.set_ylabel(DELTA_C)

    size_handles = [plt.scatter([], [], s=size(n * 1e6), color="grey", edgecolors="black", linewidths=0.5,
                                label=f"{n}M") for n in (1, 50, 100)]
    size_legend = axes[0].legend(handles=size_handles, title="Model Size", loc="upper right", labelspacing=0.7)
    axes[0].legend(handles=[mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2, alpha=0.6,
                                          label=r"100\% baseline")], loc="upper left")
    axes[0].add_artist(size_legend)

    fig.tight_layout()
    boxes = [ax.get_position() for ax in axes]
    y0, y1 = min(b.y0 for b in boxes), max(b.y1 for b in boxes)
    cax = fig.add_axes([max(b.x1 for b in boxes) + 0.012, y0, 0.01, y1 - y0])
    fig.colorbar(sc, cax=cax, label=DELTA_C)
    save(fig, f"complexity_per_plane_100models_{planes}planes_8bit_{tag(p)}")


def per_layer(p, model):
    """Per-layer ratio on the two most significant planes (P7, P6) of an 8-bit quantization."""
    layer_ratios = load(f"results/complexity_per_layer_5models{suffix(p)}.json")[model]
    n = len(layer_ratios)
    x = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(4.5, 2.2) if n <= 30 else (9.0, 2.4))
    for j, plane in enumerate([7, 6]):
        ax.bar(x + (j - 0.5) * 0.4, [r[plane] for r in layer_ratios.values()], width=0.4,
               color=PALETTE[j], label=f"P{plane}", zorder=3)
    step = 1 if n <= 30 else 2
    ax.set_xticks(x[::step])
    ax.set_xticklabels(x[::step], fontsize=7)
    ax.set_xlim(0.3, n + 0.7)
    ax.set(xlabel="Layer Index", ylabel=DELTA_C)
    if n <= 30:
        ax.legend(title="Bit-plane", loc="upper right")
    else:  # wide plots: bars reach the top-right corner, so put the legend outside
        ax.legend(title="Bit-plane", loc="upper left", bbox_to_anchor=(1.01, 1.0))
    apply_panel_style(ax)
    fig.tight_layout()
    save(fig, f"complexity_per_layer_{model}_8bit_{tag(p)}")


def vs_accuracy(p, bits=16, n_planes=16, ptq_path="results/ptq.json"):
    """Left: per-plane ratio of the 5 models (top n_planes planes). Right: PTQ top-1 accuracy vs. FP32."""
    data = load(f"results/complexities_5models_bd{bits}{suffix(p)}.json")
    ptq = load(ptq_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=two_panel_size())

    for model, color in zip(data["models"], PALETTE):
        r = ratios(data, model, bits)[-n_planes:][::-1]  # MSB first
        ax1.plot(range(n_planes), r, marker="o", markersize=5, linewidth=1.5, label=NAMES[model], color=color)
    ax1.axhline(100, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set(xlabel="Plane Index", ylabel=DELTA_C)
    ax1.set_xticks(range(n_planes))
    ax1.set_xticklabels([f"{i} \n(MSB)" if i == n_planes - 1 else f"{i} \n(LSB)" if i == 0 else f"${i}$"
                         for i in range(n_planes - 1, -1, -1)])
    ax1.legend(loc="lower right", markerfirst=False)

    bit_widths = ptq["bit_depths"]
    for model, color in zip(data["models"], PALETTE):
        d = ptq["models"][model]
        ax2.plot(bit_widths, [d["ptq"][str(b)] for b in bit_widths], color=color, linewidth=2, marker="o",
                 markersize=5, label=NAMES[model])
        ax2.axhline(d["fp32"], color=color, linewidth=1, linestyle="--", alpha=0.6)
    ax2.set(xlabel="Bit Width (PTQ)", ylabel=r"Top-1 Accuracy (\%)", xticks=bit_widths, ylim=(-0.03, None))
    ax2.set_xticklabels([f"{b}-bit" for b in bit_widths])
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y * 100:.0f}"))
    ax2.legend(handles=[mlines.Line2D([], [], color="gray", linestyle="--", label="FP32")], loc="lower right")

    for ax in (ax1, ax2):
        apply_panel_style(ax)
    fig.tight_layout()
    save(fig, f"complexity_vs_accuracy_5models_{bits}bit_{tag(p)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("figure", choices=["per_plane", "per_layer", "vs_accuracy", "all"])
    parser.add_argument("--percentile", default="99.9", type=lambda s: None if s.lower() == "none" else float(s),
                        help='clipping percentile of the results to plot, or "none" (default: 99.9)')
    parser.add_argument("--planes", type=int, choices=[4, 8], default=4, help="per_plane: show 4 or all 8 planes")
    parser.add_argument("--model", default="resnet18", help="per_layer: model name")
    parser.add_argument("--bits", type=int, default=16, help="vs_accuracy: bit depth of the results")
    args = parser.parse_args()

    if args.figure == "all":  # every figure for every clipping setting
        jobs = [(p, fn, kwargs) for p in (None, 99.99, 99.9) for fn, kwargs in
                [(per_plane, dict(planes=4)), (per_plane, dict(planes=8)), (per_layer, dict(model="resnet18")),
                 (per_layer, dict(model="resnet50")), (vs_accuracy, {})]]
    else:
        fn = {"per_plane": per_plane, "per_layer": per_layer, "vs_accuracy": vs_accuracy}[args.figure]
        kwargs = {"per_plane": dict(planes=args.planes), "per_layer": dict(model=args.model),
                  "vs_accuracy": dict(bits=args.bits)}[args.figure]
        jobs = [(args.percentile, fn, kwargs)]

    use_pedram_style()
    for p, fn, kwargs in jobs:
        try:
            fn(p, **kwargs)
        except FileNotFoundError as e:  # results for this setting haven't been computed (yet)
            print(f"Skipping {fn.__name__} ({tag(p)}): {e.filename} not found")
