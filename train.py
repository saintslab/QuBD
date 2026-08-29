import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from carbontracker.tracker import CarbonTracker
import timm
import pdb
import os
import json
import numpy as np
from utils.models import SimpleMLP
from utils.plotter import make_main_figure
from qbdm.qbdm import measure_complexity, measure_compression, measure_bitplane_compression, shuffled_weights, multi_plane_ratio, per_plane_ratio
from quantizer.quantizer import UniformSymmetricQuantizer
from quantizer.utils_quantization import attach_weight_quantizers, toggle_quantization
from tqdm import tqdm

# Environmental configuration for offline local cache priority
CACHE_DIR = os.path.expanduser("~/.cache/timm_models")
RESULTS_DIR = os.path.expanduser("./results/")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.environ['TORCH_HOME'] = CACHE_DIR
os.environ['HF_HOME'] = CACHE_DIR
os.environ['TRANSFORMERS_OFFLINE'] = '1'

### Global Experiment Parameters
USE_FASHION_MLP = True
VIT_MODEL_NAME = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k'
MLP_SCALES = [0.5,1.0,2.0]
BATCH_SIZE = 128
TRAIN_EPOCHS = 101 
BIT_DEPTHS = [8] 
DATA_BUDGETS = [100, 200]#, 500, 2000, 5000, 10000, 20000, 40000]
VAL_SIZE = 10000
REPEATS = 3
LOG_INTERVAL = 5
TRACK_HISTORY = False  # log per-epoch loss/BDM/LZMA history (for the largest DATA_BUDGETS run); set False to skip and speed up training

# Quantization-Aware Training Settings
QAT = False
QAT_BIT = 8

# Robust Normalization Parameters
USE_ROBUST_NORM = True   
ROBUST_PERCENTILE = 99.9 

def train_and_evaluate(model, budget, device, train_dataset, val_loader, bit_depths=[8], epochs=3, track_history=False, qat=False, fname=None):
    """Trains model and performs joint algorithmic, statistical, and per-plane redundancy analysis."""
    indices = torch.randperm(len(train_dataset))[:budget]
    subset = Subset(train_dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)
    
    optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()
    
    if qat:
        toggle_quantization(model, enabled=True)
        
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'sav_bdm': {bd: [] for bd in bit_depths},
        'sav_lzma': {bd: [] for bd in bit_depths},
        # Per-plane ratio (LSB index 0 -> MSB index bd-1), one list per logged epoch. This is
        # the evidence for "where" reduction is concentrated (paper Fig. 1/6/7) -- sav_bdm
        # alone dilutes that once several planes are near-random (paper Fig. 7 discussion).
        'multi_per_plane': {bd: [] for bd in bit_depths}
    }
    
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(loader)
        
        if track_history and (epoch % LOG_INTERVAL == 0):
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    total_val_loss += criterion(output, target).item()
            
            avg_val_loss = total_val_loss / len(val_loader)
 
            if qat:
                toggle_quantization(model, enabled=True)
            
            c_bin, c_multi, _ = measure_complexity(model, bit_depths=bit_depths, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
            c_bit_comp = measure_bitplane_compression(model, bit_depths=bit_depths)
            # "True structure" baseline: shuffle this same snapshot's own weights (same value
            # distribution, no spatial pattern) rather than compare against a separate
            # untrained-init model.
            with shuffled_weights(model):
                _, s_multi, _ = measure_complexity(model, bit_depths=bit_depths, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE)
            history['epochs'].append(epoch)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            for bd in bit_depths:
                history['sav_bdm'][bd].append(multi_plane_ratio(c_multi[bd], s_multi[bd]))
                history['multi_per_plane'][bd].append(per_plane_ratio(c_multi[bd], s_multi[bd]))
                history['sav_lzma'][bd].append(100 - c_bit_comp[bd]['lzma'])
            
    model.eval()
    if qat:
        toggle_quantization(model, enabled=True)
        
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = 100. * correct / len(val_loader.dataset)
    
    c_bin, c_multi_dict, _ = measure_complexity(
        model, bit_depths=bit_depths, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE
    )
    c_comp = measure_compression(model)
    c_bit_comp = measure_bitplane_compression(model, bit_depths=bit_depths)

    # "True structure" baseline for the final trained model: same weights, shuffled.
    with shuffled_weights(model):
        s_bin, s_multi_dict, _ = measure_complexity(
            model, bit_depths=bit_depths, robust=USE_ROBUST_NORM, percentile=ROBUST_PERCENTILE
        )
        s_comp = measure_compression(model)

    return c_bin, c_multi_dict, c_comp, c_bit_comp, accuracy, history, s_bin, s_multi_dict, s_comp

def main(MLP_SCALE=1.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if USE_FASHION_MLP:
        print(f"Loading FashionMNIST dataset for MLP study (Scale: {MLP_SCALE})...")
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        initial_model = SimpleMLP(scale=MLP_SCALE).to(device)
        active_model_name = f"MLP_width_{str(MLP_SCALE).replace('.', 'p')}"
    else:
        print(f"Loading CIFAR-10 dataset for {VIT_MODEL_NAME}...")
        transform = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        initial_model = timm.create_model(VIT_MODEL_NAME, pretrained=False, num_classes=10, in_chans=3).to(device)
        active_model_name = VIT_MODEL_NAME

    if QAT:
        attach_weight_quantizers(
            model=initial_model,
            exclude_layers=[],
            quantizer=UniformSymmetricQuantizer(bit_width=QAT_BIT),
            enabled=True
        )
        toggle_quantization(initial_model, enabled=True)

    val_loader = DataLoader(Subset(val_dataset, range(VAL_SIZE)), batch_size=BATCH_SIZE, shuffle=False)
    
    agg_bin, agg_acc = {b: [] for b in DATA_BUDGETS}, {b: [] for b in DATA_BUDGETS}
    agg_multi = {b: {bd: [] for bd in BIT_DEPTHS} for b in DATA_BUDGETS}
    agg_multi_per_plane = {b: {bd: [] for bd in BIT_DEPTHS} for b in DATA_BUDGETS}
    agg_gzip, agg_lzma = {b: [] for b in DATA_BUDGETS}, {b: [] for b in DATA_BUDGETS}
    agg_bit_gzip = {b: {bd: [] for bd in BIT_DEPTHS} for b in DATA_BUDGETS}
    agg_bit_lzma = {b: {bd: [] for bd in BIT_DEPTHS} for b in DATA_BUDGETS}
    
    save_name = f"{active_model_name.replace('.', '_')}"
    if QAT:
        save_name += f"_QAT{QAT_BIT}"
    else:
        save_name += "_Standard"
        
    export_data = {"metadata": {"model": active_model_name, "bit_depths": BIT_DEPTHS, "repeats": REPEATS, "qat": QAT, "qat_bit": QAT_BIT}, "results": {}}
    
    final_budget_histories = []

    for budget in DATA_BUDGETS:
        print(f"\n--- Training Study: Budget = {budget} samples ---")
        is_last_budget = TRACK_HISTORY and (budget == DATA_BUDGETS[-1])
        
        for r in range(REPEATS):
            model = SimpleMLP(scale=MLP_SCALE).to(device) if USE_FASHION_MLP else timm.create_model(VIT_MODEL_NAME, pretrained=False, num_classes=10, in_chans=3).to(device)
            num_param = sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6 
            export_data["metadata"]["num_param"] = num_param
            print(f"Num. of trainable param: {num_param:.3f} M")
            
            if QAT:
                attach_weight_quantizers(
                    model=model,
                    exclude_layers=[],
                    quantizer=UniformSymmetricQuantizer(bit_width=QAT_BIT),
                    enabled=True
                )
            
            model.load_state_dict(initial_model.state_dict())
            
            c_bin, c_multi, c_comp, c_bit_comp, acc, hist, s_bin, s_multi_dict, s_comp = train_and_evaluate(
                model, budget, device, train_dataset, val_loader,
                bit_depths=BIT_DEPTHS, epochs=TRAIN_EPOCHS,
                track_history=is_last_budget, qat=QAT, fname=save_name+'_data_'+repr(budget)+'_repeat_'+repr(r)
            )

            # Normalized against this model's own shuffled weights (a "true structure"
            # baseline), not against a separately-initialized untrained model.
            agg_bin[budget].append((c_bin/s_bin)*100)
            agg_acc[budget].append(acc)
            agg_gzip[budget].append((c_comp['gzip']/s_comp['gzip'])*100)
            agg_lzma[budget].append((c_comp['lzma']/s_comp['lzma'])*100)
            for bd in BIT_DEPTHS:
                #pdb.set_trace()
                agg_multi[budget][bd].append(multi_plane_ratio(c_multi[bd], s_multi_dict[bd]))
                agg_multi_per_plane[budget][bd].append(per_plane_ratio(c_multi[bd], s_multi_dict[bd]))
                agg_bit_gzip[budget][bd].append(c_bit_comp[bd]['gzip'])
                agg_bit_lzma[budget][bd].append(c_bit_comp[bd]['lzma'])
            
            if is_last_budget:
                final_budget_histories.append(hist)
            
            print(f"Repeat {r+1}: Acc={acc:.2f}%, BDM={agg_bin[budget][-1]:.2f}% of shuffled-self, "
                  f"QuBD({BIT_DEPTHS[-1]}b)={agg_multi[budget][BIT_DEPTHS[-1]][-1]:.2f}% of shuffled-self, "
                  f"LZMA(8b) Plane Sav={c_bit_comp[BIT_DEPTHS[-1]]['lzma']:.2f}%")

        export_data["results"][str(budget)] = {
            "acc": agg_acc[budget], "sav_bin": agg_bin[budget],
            "sav_multi": {bd: agg_multi[budget][bd] for bd in BIT_DEPTHS},
            # Per-repeat, per-plane ratio (LSB idx 0 -> MSB idx bd-1) of the final trained
            # model -- the evidence for where reduction concentrates; sav_multi above dilutes
            # that once several planes are near-random (see multi_plane_ratio docstring).
            "sav_multi_per_plane": {bd: agg_multi_per_plane[budget][bd] for bd in BIT_DEPTHS},
            "sav_gzip": agg_gzip[budget], "sav_lzma": agg_lzma[budget],
            "sav_bit_gzip": {bd: agg_bit_gzip[budget][bd] for bd in BIT_DEPTHS},
            "sav_bit_lzma": {bd: agg_bit_lzma[budget][bd] for bd in BIT_DEPTHS}
        }
        with open(RESULTS_DIR+f"{save_name}_data.json", "w") as f: json.dump(export_data, f, indent=4)

    if final_budget_histories:
        with open(RESULTS_DIR+f"{save_name}_history.json", "w") as f: json.dump(final_budget_histories, f, indent=4)

if __name__ == "__main__":
    tracker = CarbonTracker(epochs=1,
        log_dir=RESULTS_DIR,monitor_epochs=-1)
    tracker.epoch_start()
    for scale in MLP_SCALES:
        ### Run all experiments 
        main(scale)
        ### Make paper plots
        make_main_figure(RESULTS_DIR)
    tracker.epoch_end()
    tracker.stop()
