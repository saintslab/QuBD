import json
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from matplotlib.lines import Line2D
import pdb

# Set global publication-quality styles for scientific reporting
plt.rcParams.update({
    'font.size': 14,
    'lines.linewidth': 4,
    'axes.linewidth': 2,
    'lines.markersize': 10,
    'xtick.major.width': 2,
    'ytick.major.width': 2,
    'legend.fontsize': 14,
    'axes.labelsize': 24
})


model_sizes = {'0.5':0.567434,'1.0':1.462538, '2.0':4.235786, '0.25': 0.242762, '4.0': 13.714442}

def load_and_process_results(filepath):
    """
    Loads JSON data and computes statistical aggregates for accuracy 
    and multi-resolution complexity savings across budgets.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    budgets = sorted([int(k) for k in data['results'].keys()])
    str_budgets = [str(b) for b in budgets]
    
    bit_depths = data['metadata'].get('bit_depths', [])
    if not bit_depths and 'bit_depth' in data['metadata']:
        bit_depths = [data['metadata']['bit_depth']]
    
    str_bit_depths = [str(bd) for bd in bit_depths]
    
    stats = {
        'budgets': budgets,
        'acc_mean': [], 'acc_std': [],
        'bin_mean': [], 'bin_std': [],
        'gzip_mean':[], 'gzip_std':[],
        'lzma_mean':[], 'lzma_std':[],
        'multi': {bd: {'mean': [], 'std': []} for bd in str_bit_depths}
    }
    
    for b in str_budgets:
        res = data['results'][b]
        stats['acc_mean'].append(np.mean(res['acc']))
        stats['acc_std'].append(np.std(res['acc']))
        stats['bin_mean'].append(np.mean(res['sav_bin']))
        stats['bin_std'].append(np.std(res['sav_bin']))
        stats['lzma_mean'].append(np.mean(res['sav_lzma']))
        stats['lzma_std'].append(np.std(res['sav_lzma']))
        stats['gzip_mean'].append(np.mean(res['sav_gzip']))
        stats['gzip_std'].append(np.std(res['sav_gzip']))
      
        for bd in str_bit_depths:
            if isinstance(res['sav_multi'], dict):
                depth_vals = res['sav_multi'].get(bd, [])
            else:
                depth_vals = res['sav_multi']
                
            stats['multi'][bd]['mean'].append(np.mean(depth_vals))
            stats['multi'][bd]['std'].append(np.std(depth_vals))
        
    return stats, data['metadata'], str_bit_depths

def make_main_figure(RESULTS_DIR='./results/'):
    json_files = glob.glob(RESULTS_DIR+'*data.json')
    if not json_files:
        print("Error: No JSON result files discovered.")
        return

    all_data = []
    union_bit_depths = set()
    unique_budgets = set()

    for filepath in sorted(json_files):
        try:
            stats, meta, depths = load_and_process_results(filepath)
            label = os.path.splitext(os.path.basename(filepath))[0].split('_budget')[0].replace('p','.').replace('_',' ')
            capacity = meta.get('baseline_multi', 0)
            if isinstance(capacity, dict):
                max_depth = str(max([int(d) for d in capacity.keys()]))
                capacity = capacity[max_depth]
                
            all_data.append({
                'label': label, 
                'capacity': capacity, 
                'stats': stats, 
                'meta': meta, 
                'depths': depths
            })
            for d in depths:
                union_bit_depths.add(d)
            for b in stats['budgets']:
                unique_budgets.add(b)
        except Exception as e:
            print(f"Skipping {filepath}: {e}")

    all_data.sort(key=lambda x: x['capacity'])
    sorted_depths = sorted(list(union_bit_depths), key=lambda x: int(x))
    sorted_budgets = sorted(list(unique_budgets))
    max_b = np.max(sorted_budgets)
    def get_sizes(budgets):
        return (np.array(budgets) / max_b) * 800 + 100

    cmap = plt.get_cmap('tab10')
    b_cmap = plt.get_cmap('viridis')

    # --- FIGURE 4: ACCURACY-COMPLEXITY PARETO ---
    fig_p_bin, ax_p_bin = plt.subplots(figsize=(8,6))
    for i, model in enumerate(all_data):
        sizes = get_sizes(stats['budgets'])
        label, stats, color = model['label'], model['stats'], cmap(i % 10)
        size = model_sizes[label.split('width')[1].split(' ')[1]]
        label = f'{size:.1f}M'

        ax_p_bin.scatter(stats['bin_mean'], stats['acc_mean'], s=sizes, color=color, edgecolors='white', alpha=0.5, zorder=5)
        ax_p_bin.errorbar(stats['bin_mean'], stats['acc_mean'], xerr=stats['bin_std'], yerr=stats['acc_std'], fmt='s', color=color, label=label, lw=4, capsize=5,alpha=0.5)
        ax_p_bin.errorbar(stats['bin_mean'], stats['acc_mean'], xerr=stats['bin_std'], yerr=stats['acc_std'], fmt='-', color=color, lw=1, capsize=5)

    ax_p_bin.set_xlabel('BDM Complexity Reduction (%)'); ax_p_bin.set_ylabel('Accuracy (%)')
    ax_p_bin.set_xlim([-5,105])
    ax_p_bin.legend()
    # Model Legend
    leg2 = ax_p_bin.legend(loc='lower center', title="# Param.")
    
    # Size Legend (using squares for consistency with multi-bit markers)
    size_handles_bin = [Line2D([0], [0], marker='o', color='w', label=f'{b}',alpha=0.5, 
                                 markerfacecolor='gray', markersize=np.sqrt(s)) 
                          for b, s in zip(sorted_budgets, sizes)]
    ax_p_bin.add_artist(leg2)
    ax_p_bin.legend(handles=size_handles_bin, loc='lower right', title="# Samples", frameon=True)


    fig_p_bin.savefig(RESULTS_DIR+'pareto_binary.pdf', bbox_inches='tight')

    # --- FIGURE 5: ACCURACY-COMPRESSION PARETO ---
    fig_p_bin, ax_p_bin = plt.subplots(figsize=(8,6))
    for i, model in enumerate(all_data):
        sizes = get_sizes(stats['budgets'])
        label, stats, color = model['label'], model['stats'], cmap(i % 10)
        size = model_sizes[label.split('width')[1].split(' ')[1]]
        #label = label.split('width')[0]+f'({size:.1f}M)'
        label = f'{size:.1f}M'

        ax_p_bin.scatter(stats['lzma_mean'], stats['acc_mean'], s=sizes, color=color, edgecolors='white', alpha=0.5, zorder=5)
        ax_p_bin.errorbar(stats['lzma_mean'], stats['acc_mean'], xerr=stats['lzma_std'], yerr=stats['acc_std'], fmt='s', color=color, label=label, lw=4, capsize=5)
        ax_p_bin.errorbar(stats['lzma_mean'], stats['acc_mean'], xerr=stats['lzma_std'], yerr=stats['acc_std'], fmt='-', color=color, lw=1)

    #ax_p_bin.set_title('Accuracy-Compression Pareto: LZMA', fontsize=22)
    ax_p_bin.set_xlabel('LZMA Complexity Reduction (%)'); ax_p_bin.set_ylabel('Accuracy (%)')
    #ax_p_bin.grid(True, alpha=0.3); 
    ax_p_bin.set_xlim([-5,105])
    ax_p_bin.legend()
    # Model Legend
    leg2 = ax_p_bin.legend(loc='lower center', title="# Param.")
    
    # Size Legend (using squares for consistency with multi-bit markers)
    size_handles_bin = [Line2D([0], [0], marker='o', color='w', label=f'{b}',alpha=0.5, 
                                 markerfacecolor='gray', markersize=np.sqrt(s)) 
                          for b, s in zip(sorted_budgets, sizes)]
    ax_p_bin.add_artist(leg2)
    ax_p_bin.legend(handles=size_handles_bin, loc='lower right', title="# Samples", frameon=True)


    fig_p_bin.savefig(RESULTS_DIR+'pareto_lzma.pdf', bbox_inches='tight')

    # --- FIGURE 5: ACCURACY-COMPRESSION PARETO ---
    fig_p_bin, ax_p_bin = plt.subplots(figsize=(8,6))
    for i, model in enumerate(all_data):
        sizes = get_sizes(stats['budgets'])
        label, stats, color = model['label'], model['stats'], cmap(i % 10)
        size = model_sizes[label.split('width')[1].split(' ')[1]]
        #label = label.split('width')[0]+f'({size:.1f}M)'
        label = f'{size:.1f}M'

        ax_p_bin.scatter(stats['gzip_mean'], stats['acc_mean'], s=sizes, color=color, edgecolors='white', alpha=0.5, zorder=5)
        ax_p_bin.errorbar(stats['gzip_mean'], stats['acc_mean'], xerr=stats['gzip_std'], yerr=stats['acc_std'], fmt='s', color=color, label=label, lw=4, capsize=5,alpha=0.5)
        ax_p_bin.errorbar(stats['gzip_mean'], stats['acc_mean'], xerr=stats['gzip_std'], yerr=stats['acc_std'], fmt='-', color=color, lw=1, alpha=0.5)

    #ax_p_bin.set_title('Accuracy-Compression Pareto: LZMA', fontsize=22)
    ax_p_bin.set_xlabel('GZIP Complexity Reduction (%)'); ax_p_bin.set_ylabel('Accuracy (%)')
    #ax_p_bin.grid(True, alpha=0.3); 
    ax_p_bin.legend()
    ax_p_bin.set_xlim([-5,105])
   # Model Legend
    leg2 = ax_p_bin.legend(loc='lower center', title="# Param.")
    
    # Size Legend (using squares for consistency with multi-bit markers)
    size_handles_bin = [Line2D([0], [0], marker='o', color='w', label=f'{b}',alpha=0.5, 
                                 markerfacecolor='gray', markersize=np.sqrt(s)) 
                          for b, s in zip(sorted_budgets, sizes)]
    ax_p_bin.add_artist(leg2)
    ax_p_bin.legend(handles=size_handles_bin, loc='lower right', title="# Samples", frameon=True)


    fig_p_bin.savefig(RESULTS_DIR+'pareto_gzip.pdf', bbox_inches='tight')



    fig_p_multi, ax_p_multi = plt.subplots(figsize=(8,6))
    for i, model in enumerate(all_data):
        label, stats, color = model['label'], model['stats'], cmap(i % 10)
        size = model_sizes[label.split('width')[1].split(' ')[1]]
        #label = label.split('width')[0]+f'({size:.1f}M)' 
        label = f'{size:.1f}M'

        max_bd = model['depths'][-2]
        m_mean, m_std = stats['multi'][max_bd]['mean'], stats['multi'][max_bd]['std']
        ax_p_multi.scatter(m_mean, stats['acc_mean'], s=sizes, color=color, edgecolors='white', alpha=0.5, zorder=5)
        ax_p_multi.errorbar(m_mean, stats['acc_mean'], xerr=m_std, yerr=stats['acc_std'], fmt='s', color=color, label=label, lw=4, capsize=5,alpha=0.5)
        ax_p_multi.errorbar(m_mean, stats['acc_mean'], xerr=m_std, yerr=stats['acc_std'], fmt='-', color=color, lw=1)
 
    #ax_p_multi.set_title('Accuracy-Complexity Pareto: Multi-bit BDM', fontsize=22)
    ax_p_multi.set_xlabel('QuBD Complexity Reduction (%)'); ax_p_multi.set_ylabel('Accuracy (%)')
    #ax_p_multi.grid(True, alpha=0.3); 
    ax_p_multi.set_xlim([-5,105])
    ax_p_multi.legend()

    # Model Legend
    leg2 = ax_p_multi.legend(loc='lower center', title="# Param.")
    
    # Size Legend (using squares for consistency with multi-bit markers)
    size_handles_multi = [Line2D([0], [0], marker='o', color='w', label=f'{b}', alpha=0.5,
                                 markerfacecolor='gray', markersize=np.sqrt(s)) 
                          for b, s in zip(sorted_budgets, sizes)]
    ax_p_multi.add_artist(leg2)
    ax_p_multi.legend(handles=size_handles_multi, loc='lower right', title="# Samples", frameon=True)


    fig_p_multi.savefig(RESULTS_DIR+'pareto_multi_bit.pdf', bbox_inches='tight')
    print("Done!")


def make_pythia_dynamics_figure(results_path, out_dir='./results/'):
    """Plots training loss (from wandb) alongside weight complexity reduction (from qbdm)
    across a sweep of Pythia training-step checkpoints."""
    with open(results_path, 'r') as f:
        data = json.load(f)

    meta, hist = data['metadata'], data['history']
    steps = np.array(hist['steps'])
    max_bd = str(max(meta['bit_depths']))

    valid = [(s, l) for s, l in zip(steps, hist['train_loss']) if l is not None]
    loss_steps, loss_vals = zip(*valid)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xscale('symlog', linthresh=1)

    color_loss = 'tab:red'
    ax1.plot(loss_steps, loss_vals, marker='o', color=color_loss, label='Train loss (wandb)')
    ax1.set_xlabel('Training step')
    ax1.set_ylabel('Train Loss', color=color_loss)
    ax1.tick_params(axis='y', labelcolor=color_loss)

    val_loss = meta.get('val_loss')
    if val_loss and val_loss.get('steps'):
        ax1.scatter(val_loss['steps'], val_loss['loss'], marker='*', s=220, color='black',
                    edgecolors='white', zorder=6, label='Validation loss (wandb, sparse)')

    ax2 = ax1.twinx()
    color_bdm, color_lzma, color_msb = 'tab:blue', 'tab:green', 'tab:purple'
    ax2.plot(steps, hist['sav_bin'], marker='s', color=color_bdm, label='BDM complexity reduction (%)')
    ax2.plot(steps, hist['sav_lzma'][max_bd], marker='^', color=color_lzma,
              label=f'LZMA {max_bd}-bit plane savings (%)')
    if 'sav_msb' in hist:
        ax2.plot(steps, hist['sav_msb'][max_bd], marker='d', color=color_msb,
                  label=f'BDM MSB-plane (bit {int(max_bd) - 1}) reduction (%)')
    ax2.set_ylabel('Complexity Reduction (%)')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    model_name = meta['model'].split('/')[-1]
    ax1.set_title(f'{model_name}: Training Loss vs. Weight Complexity (reduction vs. step0)')
    fig.tight_layout()

    save_path = os.path.join(out_dir, f'{model_name}_dynamics.pdf')
    fig.savefig(save_path, bbox_inches='tight')
    print(f"Saved figure to {save_path}")
    return save_path


