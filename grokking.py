import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from qbdm.qbdm import measure_complexity, shuffled_weights, multi_plane_ratio, per_plane_ratio
import json

RESULTS_DIR = os.path.expanduser("./results/")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set global publication-quality styles
plt.rcParams.update({
    'font.size': 18,
    'lines.linewidth': 3,
    'axes.linewidth': 2,
    'legend.fontsize': 14,
    'axes.labelsize': 18
})

# 1. Configuration & Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Executing on: {device}")

# Repeat parameter
NUM_REPEATS = 1

# Training Hyperparameters
USE_ROBUST_NORM = True
ROBUST_PERCENTILE = 99.0
BIT_WIDTH = 8
# If set (e.g. 8), the quantizer's dynamic range is computed per BLOCK_SIZE x BLOCK_SIZE
# block of each weight tensor instead of once globally -- localizes the effect of the
# quantizer's own range on the measured complexity. None = original global-range behavior.
# See qbdm.qbdm.get_bitplanes()'s docstring for details.
BLOCK_SIZE = None
P = 97
D_MODEL = 512
LR = 1e-3
WD = 1
TRAIN_FRACTION = 0.5
NUM_EPOCHS = 15000
EVAL_FREQ = 100

# Sparsity tracking: fraction of eligible weight elements whose |w| has fallen below
# SPARSITY_REL_EPS times that *same tensor's own* initial std (captured once per run,
# before training). A per-tensor RELATIVE threshold rather than one fixed absolute value,
# since embed/fc1/fc2 start at very different scales (nn.Embedding's default init has
# std~1, nn.Linear's default init is ~1/sqrt(fan_in) ~ 0.03-0.04) -- a single absolute
# epsilon would call one layer "mostly sparse" at init and another "never sparse", which
# isn't meaningful. This is a cheap diagnostic for the "is qbdm_self's decline just
# weight-decay-driven pruning of task-irrelevant weights toward ~0" hypothesis: if
# sparsity climbs in lockstep with qbdm_self's decline, that's consistent with pruning;
# if qbdm_self moves well beyond what sparsity alone would predict, that points to
# something more than pruning.
SPARSITY_REL_EPS = 0.01

class GrokkingModel(nn.Module):
    def __init__(self, p, d):
        super().__init__()
        self.embed = nn.Embedding(p, d)
        self.fc1 = nn.Linear(d * 2, d)
        self.fc2 = nn.Linear(d, p)

    def forward(self, x):
        z = self.embed(x).view(x.shape[0], -1)
        return self.fc2(torch.relu(self.fc1(z)))


def sparsity_fraction(model, init_std_by_name, rel_eps=SPARSITY_REL_EPS):
    """Fraction of eligible weight elements (trainable, dim>=2 -- same selection criteria
    as qbdm.qbdm._eligible_weight_params) with |w| < rel_eps * that tensor's own initial
    std, pooled across all eligible tensors."""
    below, total = 0, 0
    for name, p in model.named_parameters():
        if name in init_std_by_name and p.requires_grad and p.dim() >= 2:
            thresh = rel_eps * init_std_by_name[name]
            below += (p.data.abs() < thresh).sum().item()
            total += p.data.numel()
    return below / total if total > 0 else 0.0

# Data structure to hold all runs
# We use lists to collect data, then convert to numpy for statistics
all_runs = {
    't_acc': [], 'v_acc': [],
    't_loss': [], 'v_loss': [],
    'qbdm_rand': [], 'qbdm_self': [],
    'qbdm_per_plane_rand': [], 'qbdm_per_plane_self': [],
    'sparsity': []
}
steps = None

# 2. Execution Loop
for run in range(NUM_REPEATS):
    print(f"\n--- Starting Run {run+1}/{NUM_REPEATS} ---")

    # Re-initialize model and data split for each run
    model = GrokkingModel(P, D_MODEL).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()

    # Random-init baseline (paper's Eq. 14 f_RND): this run's freshly-initialized model,
    # measured once before any training step and reused as the fixed denominator for every
    # eval this run (each run draws its own fresh init, so unlike train.py this can't be
    # hoisted out of the run loop).
    _, b_dict, _ = measure_complexity(model, bit_depths=[BIT_WIDTH],
                                       robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE,
                                       block_size=BLOCK_SIZE)

    # Per-tensor initial std, captured once before any training step -- the fixed reference
    # scale for sparsity_fraction()'s relative threshold (see SPARSITY_REL_EPS comment above).
    init_std_by_name = {name: p.data.std().item() for name, p in model.named_parameters()
                         if p.requires_grad and p.dim() >= 2}

    x_data = torch.cartesian_prod(torch.arange(P), torch.arange(P)).to(device)
    y_data = ((x_data[:, 0] + x_data[:, 1]) % P).to(device)
    indices = torch.randperm(P * P)
    split_idx = int(TRAIN_FRACTION * P * P)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    run_history = {'t_acc': [], 'v_acc': [], 't_loss': [], 'v_loss': [],
                    'qbdm_rand': [], 'qbdm_self': [],
                    'qbdm_per_plane_rand': [], 'qbdm_per_plane_self': [],
                    'sparsity': []}
    current_steps = []

    for epoch in range(NUM_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()

        t_logits = model(x_data[train_idx])
        t_loss = criterion(t_logits, y_data[train_idx])
        t_loss.backward()
        optimizer.step()

        if epoch % EVAL_FREQ == 0:
            model.eval()
            with torch.no_grad():
                v_logits = model(x_data[val_idx])
                v_loss = criterion(v_logits, y_data[val_idx])

                t_acc = (t_logits.argmax(1) == y_data[train_idx]).float().mean().item()
                v_acc = (v_logits.argmax(1) == y_data[val_idx]).float().mean().item()
                _, c_dict, _ = measure_complexity(model, bit_depths=[BIT_WIDTH],
                                              robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE,
                                              block_size=BLOCK_SIZE)
                # Self-shuffle baseline: this same snapshot's own weights, randomly permuted
                # (same value distribution, no spatial pattern). Reported alongside the
                # random-init baseline (b_dict, measured once above) rather than replacing it
                # -- see train.py/qbdm.py multi_plane_ratio() docstrings for the distinction.
                with shuffled_weights(model):
                    _, s_dict, _ = measure_complexity(model, bit_depths=[BIT_WIDTH],
                                                  robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE,
                                                  block_size=BLOCK_SIZE)
                qbdm_rand = multi_plane_ratio(c_dict[BIT_WIDTH], b_dict[BIT_WIDTH])
                qbdm_self = multi_plane_ratio(c_dict[BIT_WIDTH], s_dict[BIT_WIDTH])
                sparsity = sparsity_fraction(model, init_std_by_name)

                current_steps.append(epoch)
                run_history['t_acc'].append(t_acc)
                run_history['v_acc'].append(v_acc)
                run_history['t_loss'].append(t_loss.item())
                run_history['v_loss'].append(v_loss.item())
                run_history['qbdm_rand'].append(qbdm_rand)
                run_history['qbdm_self'].append(qbdm_self)
                run_history['sparsity'].append(sparsity)
                # Per-plane (LSB idx 0 -> MSB idx BIT_WIDTH-1) against each baseline -- where
                # reduction concentrates; the aggregates above dilute that once several planes
                # are near-random (paper Fig. 7 discussion).
                run_history['qbdm_per_plane_rand'].append(per_plane_ratio(c_dict[BIT_WIDTH], b_dict[BIT_WIDTH]))
                run_history['qbdm_per_plane_self'].append(per_plane_ratio(c_dict[BIT_WIDTH], s_dict[BIT_WIDTH]))

                if epoch % 1000 == 0:
                    print(f"Epoch {epoch:5} | Val Acc: {v_acc:.2f} | "
                          f"QBDM: {qbdm_rand:.2f}% of rand / {qbdm_self:.2f}% of self | "
                          f"Sparsity: {sparsity*100:.2f}%")

    # Store run results
    for key in all_runs:
        all_runs[key].append(run_history[key])
    steps = current_steps

# 3. Statistical Aggregation
stats = {}
for key, data in all_runs.items():
    arr = np.array(data)
    stats[f'{key}_mean'] = np.mean(arr, axis=0)
    stats[f'{key}_std'] = np.std(arr, axis=0)

# 1. Prepare data for serialization
# We iterate through the stats and convert NumPy arrays to lists
stats_to_save = {
    key: value.tolist() if hasattr(value, "tolist") else value
    for key, value in stats.items()
}

# 2. Include the step indices to preserve the temporal axis
stats_to_save['steps'] = steps

# 3. Write to JSON
file_path = os.path.join(RESULTS_DIR, 'grokking_stats.json')
with open(file_path, 'w') as f:
    json.dump(stats_to_save, f, indent=4)

print(f"Statistics successfully saved to {file_path}")

# 4. Visualization with Shaded Error Bars
fig, (ax1, ax2, ax4) = plt.subplots(3, 1, figsize=(12, 18), sharex=True)

def plot_with_std(ax, x, mean, std, label, color, linestyle='-',linewidth=4):
    ax.plot(x, mean, label=label, color=color, linestyle=linestyle,linewidth=linewidth)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)

# Accuracy Plot
plot_with_std(ax1, steps, stats['t_acc_mean'], stats['t_acc_std'], 'Train Accuracy', '#1f77b4')
plot_with_std(ax1, steps, stats['v_acc_mean'], stats['v_acc_std'], 'Val Accuracy', '#d62728', '--')
ax1.set_ylabel('Accuracy')
ax1.set_xscale('log') # This sets the logarithmic scale
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.2)
ax1.set_ylim(-0.05, 1.05)

# Loss and Complexity Plot
plot_with_std(ax2, steps, stats['t_loss_mean'], stats['t_loss_std'], 'Train Loss', '#1f77b4')
plot_with_std(ax2, steps, stats['v_loss_mean'], stats['v_loss_std'], 'Val Loss', '#d62728', '--')
ax2.set_ylabel('Loss')
ax2.set_xscale('log') # This sets the logarithmic scale
ax2.grid(True, alpha=0.2)

# Secondary Axis for QBDM -- both baselines (see multi_plane_ratio()/shuffled_weights()
# docstrings for the distinction).
ax3 = ax2.twinx()
plot_with_std(ax3, steps, stats['qbdm_rand_mean'], stats['qbdm_rand_std'], r'$\Delta C_{QuBD}$ (vs. rand)', 'tab:green')
plot_with_std(ax3, steps, stats['qbdm_self_mean'], stats['qbdm_self_std'], r'$\Delta C_{QuBD}$ (vs. self)', 'tab:olive', '--')
ax3.set_ylabel(r'$\Delta C_{QuBD}$ (%)')

# Merge legends for the middle subplot
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Sparsity Plot -- fraction of eligible weight elements with |w| < SPARSITY_REL_EPS * that
# tensor's own initial std (see SPARSITY_REL_EPS comment above). Plotted alongside qbdm_self
# on its own panel (same log-epoch x-axis) so the two trends can be compared by eye: if
# qbdm_self's decline is just weight-decay-driven pruning of task-irrelevant weights toward
# ~0, the two curves should track each other; a qbdm_self drop well beyond what sparsity
# alone would predict points to something more than pruning.
sparsity_pct_mean = np.array(stats['sparsity_mean']) * 100
sparsity_pct_std = np.array(stats['sparsity_std']) * 100
plot_with_std(ax4, steps, sparsity_pct_mean, sparsity_pct_std, 'Sparsity (% of weights)', 'tab:purple')
ax4.set_ylabel('Sparsity (%)')
ax4.set_xlabel('Epochs (log)')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.2)
ax4.legend(loc='upper left')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'grokking_averaged.pdf')
plt.savefig(plot_path)
print(f"\nProcessing complete. Plot saved as {plot_path}")
