import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import timm
import time
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from qbdm.qbdm import measure_complexity

# Environmental configuration for offline local cache priority
CACHE_DIR = os.path.expanduser("~/.cache/timm_models")
os.makedirs(CACHE_DIR, exist_ok=True)
RES_DIR = './results'
os.makedirs(RES_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_OFFLINE'] = '1'

### Global Experiment Parameters
USE_FASHION_MLP = True
VIT_MODEL_NAME = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
MLP_SCALE = 0.5         # Scale factor applied to the number of hidden dimensions
BATCH_SIZE = 64
TRAIN_EPOCHS = 100 
BIT_DEPTHS = [2, 4, 8]  # Multiple bit widths for analysis
DATA_BUDGETS = [100, 500]#, 1000, 2000, 10000, 20000, 40000]    # List of training samples to study
VAL_SIZE = 10000
REPEATS = 3

# Robust Normalization Parameters
USE_ROBUST_NORM = True   # Toggle percentile-based clamping to handle outliers
ROBUST_PERCENTILE = 99.9 # Percentile of the distribution to keep (clips 0.05% on each tail)

# Global BDM instance for workers
#bdm_instance = BDM(ndim=2, nsymbols=2)

class SimpleMLP(nn.Module):
    """A four-layered Multi-Layer Perceptron with width scaling."""
    def __init__(self, input_dim=784, hidden_dims=[1024, 512, 256], num_classes=10, scale=1.0):
        super(SimpleMLP, self).__init__()
        # Apply the scaling factor to the number of neurons in each hidden layer
        scaled_hidden_dims = [int(h * scale) for h in hidden_dims]
        self.layers = nn.ModuleList()
        curr_dim = input_dim
        for h_dim in scaled_hidden_dims:
            self.layers.append(nn.Linear(curr_dim, h_dim))
            curr_dim = h_dim
        self.final_layer = nn.Linear(curr_dim, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.layers:
            # Linear transformation followed by ReLU activation
            x = F.relu(layer(x))
        x = self.final_layer(x)
        return x

def train_and_evaluate(model, budget, device, train_dataset, val_loader, bit_depths=[8], epochs=3):
    """Trains a fresh model on a specific data budget and evaluates multi-resolution complexity."""
    indices = torch.randperm(len(train_dataset))[:budget]
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        model.train()
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(val_loader.dataset)
    
    c_bin, c_multi_dict, _ = measure_complexity(
        model, 
        bit_depths=bit_depths, 
        robust=USE_ROBUST_NORM, 
        percentile=ROBUST_PERCENTILE
    )
    return c_bin, c_multi_dict, accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if USE_FASHION_MLP:
        print(f"Loading FashionMNIST dataset for MLP study (Width Scale: {MLP_SCALE})...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        initial_model = SimpleMLP(scale=MLP_SCALE).to(device)
        dataset_name = "FashionMNIST"
        active_model_name = f"MLP_width_{str(MLP_SCALE).replace('.', 'p')}"
    else:
        print(f"Loading CIFAR-10 dataset for {VIT_MODEL_NAME}...")
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        initial_model = timm.create_model(VIT_MODEL_NAME, pretrained=False, num_classes=10, in_chans=3).to(device)
        dataset_name = "CIFAR-10"
        active_model_name = VIT_MODEL_NAME

    val_subset = Subset(val_dataset, range(VAL_SIZE))
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    agg_bin = {b: [] for b in DATA_BUDGETS}
    agg_multi = {b: {bd: [] for bd in BIT_DEPTHS} for b in DATA_BUDGETS}
    agg_acc = {b: [] for b in DATA_BUDGETS}
    
    print(f"Measuring Untrained Baseline (Robust Norm: {USE_ROBUST_NORM})...")
    b_bin, b_multi_dict, _ = measure_complexity(
        initial_model, 
        bit_depths=BIT_DEPTHS, 
        robust=USE_ROBUST_NORM, 
        percentile=ROBUST_PERCENTILE
    )
    
    save_name = RES_DIR+f"/{active_model_name.replace('.', '_')}_budget_study"
    if USE_ROBUST_NORM:
        save_name += f"_robust_{str(ROBUST_PERCENTILE).replace('.', 'p')}"

    export_data = {
        "metadata": {
            "model": active_model_name,
            "dataset": dataset_name,
            "bit_depths": BIT_DEPTHS,
            "repeats": REPEATS,
            "baseline_bin": b_bin,
            "baseline_multi": b_multi_dict,
            "mlp_width_scale": MLP_SCALE if USE_FASHION_MLP else None,
            "robust_norm": USE_ROBUST_NORM,
            "robust_percentile": ROBUST_PERCENTILE
        },
        "results": {}
    }

    for budget in DATA_BUDGETS:
        print(f"\n--- Training Study: Budget = {budget} samples ({REPEATS} repeats) ---")
        for r in range(REPEATS):
            if USE_FASHION_MLP:
                model = SimpleMLP(scale=MLP_SCALE).to(device)
            else:
                model = timm.create_model(VIT_MODEL_NAME, pretrained=False, num_classes=10, in_chans=3).to(device)
            
            model.load_state_dict(initial_model.state_dict())
            
            c_bin, c_multi_dict, acc = train_and_evaluate(
                model, budget, device, train_dataset, val_loader, 
                bit_depths=BIT_DEPTHS, epochs=TRAIN_EPOCHS
            )
            
            agg_bin[budget].append(c_bin)
            agg_acc[budget].append(acc)
            for bd in BIT_DEPTHS:
                agg_multi[budget][bd].append(c_multi_dict[bd])
            
            log_msg = f"Repeat {r+1}/{REPEATS}: Acc={acc:.2f}%, Bin-K Savings={(1 - c_bin/b_bin)*100:.2f}%"
            for bd in BIT_DEPTHS:
                savings = (1 - c_multi_dict[bd]/b_multi_dict[bd])*100
                log_msg += f", {bd}Bit-K Savings={savings:.2f}%"
            print(log_msg)

        # Update export structure incrementally
        export_data["results"][str(budget)] = {
            "sav_bin": [(1 - val/b_bin)*100 for val in agg_bin[budget]],
            "sav_multi": {bd: [(1 - val/b_multi_dict[bd])*100 for val in agg_multi[budget][bd]] for bd in BIT_DEPTHS},
            "acc": agg_acc[budget]
        }
        with open(f"{save_name}.json", "w") as f:
            json.dump(export_data, f, indent=4)
        print(f"Results for budget {budget} saved to {save_name}.json.")

    # Aggregation logic corrected to use integer 'bd' keys
    m_sav_bin = [np.mean(export_data["results"][str(b)]["sav_bin"]) for b in DATA_BUDGETS]
    s_sav_bin = [np.std(export_data["results"][str(b)]["sav_bin"]) for b in DATA_BUDGETS]
    
    m_sav_multi = {bd: [np.mean(export_data["results"][str(b)]["sav_multi"][bd]) for b in DATA_BUDGETS] for bd in BIT_DEPTHS}
    s_sav_multi = {bd: [np.std(export_data["results"][str(b)]["sav_multi"][bd]) for b in DATA_BUDGETS] for bd in BIT_DEPTHS}
    
    m_acc = [np.mean(agg_acc[b]) for b in DATA_BUDGETS]
    s_acc = [np.std(agg_acc[b]) for b in DATA_BUDGETS]

    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Complexity Reduction: {active_model_name} (Robust: {USE_ROBUST_NORM})", fontsize=16)

    ax1.errorbar(DATA_BUDGETS, m_sav_bin, yerr=s_sav_bin, fmt='g-o', capsize=5, label='Binary Savings (%)')
    ax1_twin = ax1.twinx()
    ax1_twin.errorbar(DATA_BUDGETS, m_acc, yerr=s_acc, fmt='k--x', alpha=0.3, capsize=3, label='Val Accuracy (%)')
    ax1.set_ylabel('Binary Reduction (%)', color='g')
    ax1_twin.set_ylabel('Accuracy (%)', color='k')
    ax1.set_title('Binary Sign Manifold Simplification')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(BIT_DEPTHS)))
    for i, bd in enumerate(BIT_DEPTHS):
        ax2.errorbar(DATA_BUDGETS, m_sav_multi[bd], yerr=s_sav_multi[bd], 
                     fmt='-s', color=colors[i], capsize=5, label=f'{bd}-Bit Savings')
    
    ax2_twin = ax2.twinx()
    ax2_twin.errorbar(DATA_BUDGETS, m_acc, yerr=s_acc, fmt='k--x', alpha=0.3)
    ax2.set_ylabel('Multi-bit Reduction (%)')
    ax2_twin.set_ylabel('Accuracy (%)', color='k')
    ax2.set_xlabel('Training Data Budget (Samples)')
    ax2.set_title('Hierarchical Multi-Resolution Simplification')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_name}.pdf")
    plt.show()

if __name__ == "__main__":
    main()
