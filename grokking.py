import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from qbdm.qbdm import measure_complexity, shuffled_weights, multi_plane_ratio
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
NUM_REPEATS = 3

# Training Hyperparameters
USE_ROBUST_NORM = True
ROBUST_PERCENTILE = 99.9
BIT_WIDTH = 4
P = 97
D_MODEL = 512
LR = 1e-3
WD = 1
TRAIN_FRACTION = 0.5
NUM_EPOCHS = 11000
EVAL_FREQ = 100

class GrokkingModel(nn.Module):
    def __init__(self, p, d):
        super().__init__()
        self.embed = nn.Embedding(p, d)
        self.fc1 = nn.Linear(d * 2, d)
        self.fc2 = nn.Linear(d, p)

    def forward(self, x):
        z = self.embed(x).view(x.shape[0], -1)
        return self.fc2(torch.relu(self.fc1(z)))

# Data structure to hold all runs
# We use lists to collect data, then convert to numpy for statistics
all_runs = {
    't_acc': [], 'v_acc': [],
    't_loss': [], 'v_loss': [],
    'qbdm': []
}
steps = None

# 2. Execution Loop
for run in range(NUM_REPEATS):
    print(f"\n--- Starting Run {run+1}/{NUM_REPEATS} ---")

    # Re-initialize model and data split for each run
    model = GrokkingModel(P, D_MODEL).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss()

    x_data = torch.cartesian_prod(torch.arange(P), torch.arange(P)).to(device)
    y_data = ((x_data[:, 0] + x_data[:, 1]) % P).to(device)
    indices = torch.randperm(P * P)
    split_idx = int(TRAIN_FRACTION * P * P)
    train_idx, val_idx = indices[:split_idx], indices[split_idx:]

    run_history = {'t_acc': [], 'v_acc': [], 't_loss': [], 'v_loss': [], 'qbdm': []}
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
                                              robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
                # "True structure" baseline: shuffle this same snapshot's own weights (same
                # value distribution, no spatial pattern) rather than track a raw, unnormalized
                # sum -- isolates genuine structure from whatever the value distribution alone
                # would give, same rationale as train.py. Matches the paper's own grokking
                # figure (Fig. 5, right), which plots Delta C_QuBD (%), not a raw BDM sum.
                with shuffled_weights(model):
                    _, s_dict, _ = measure_complexity(model, bit_depths=[BIT_WIDTH],
                                                  robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
                qbdm_score = multi_plane_ratio(c_dict[BIT_WIDTH], s_dict[BIT_WIDTH])

                current_steps.append(epoch)
                run_history['t_acc'].append(t_acc)
                run_history['v_acc'].append(v_acc)
                run_history['t_loss'].append(t_loss.item())
                run_history['v_loss'].append(v_loss.item())
                run_history['qbdm'].append(qbdm_score)

                if epoch % 5000 == 0:
                    print(f"Epoch {epoch:5} | Val Acc: {v_acc:.2f} | QBDM: {qbdm_score:.2f}% of shuffled-self")

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
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

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

ax2.set_xlabel('Epochs (log)')
ax2.set_xscale('log') # This sets the logarithmic scale
ax2.grid(True, alpha=0.2)

# Secondary Axis for QBDM
ax3 = ax2.twinx()
plot_with_std(ax3, steps, stats['qbdm_mean'], stats['qbdm_std'], r'$\Delta C_{QuBD}$', 'tab:green')
ax3.set_ylabel(r'$\Delta C_{QuBD}$ (%)')

# Merge legends for the bottom subplot
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, 'grokking_averaged.pdf')
plt.savefig(plot_path)
print(f"\nProcessing complete. Plot saved as {plot_path}")
