import os
import scipy.io
import numpy as np
import matplotlib.pyplot as plt

save_path = r'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_syn'

# Define data paths and labels
data_configs = [
    {
        'path': r'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\DnCNN_Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI\LI_synthesize',
        'label': 'LI+DnCNN+Attention',
        'color': '#002060',  # Dark Navy
        'marker': 'o'
    },
    {
        'path': r'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\Clipped_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LI_synthesize',
        'label': 'LI+DnCNN',
        'color': '#0070c0',  # Royal Blue
        'marker': 's'
    },
    {
        'path': r'C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\single_dataset\Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps\LS_synthesize',
        'label': 'LS+Attention',
        'color': '#c00000',  # Dark Red
        'marker': '^'
    }
]

# Set overall plot output folder
output_dir = save_path
os.makedirs(output_dir, exist_ok=True)

# 1. Load and parse synthesized results
loaded_data = []
for config in data_configs:
    mat_path = os.path.join(config['path'], 'synthesized_results.mat')
    if not os.path.exists(mat_path):
        print(f"[Warning] File not found: {mat_path}")
        continue
        
    print(f"Loading results from: {mat_path}")
    mat = scipy.io.loadmat(mat_path)
    
    # Extract metric arrays
    snr = mat['snr'].squeeze()
    mmse = mat['mmse_test'].squeeze()
    nmse_db = mat['nmse_test_db'].squeeze()
    ssim = mat['ssim_test'].squeeze()
    
    mmse_in = mat['mmse_input_test'].squeeze()
    nmse_db_in = mat['nmse_input_test_db'].squeeze()
    ssim_in = mat['ssim_input_test'].squeeze()
    
    loaded_data.append({
        'label': config['label'],
        'color': config['color'],
        'marker': config['marker'],
        'snr': snr,
        'mmse': mmse,
        'nmse_db': nmse_db,
        'ssim': ssim,
        'mmse_in': mmse_in,
        'nmse_db_in': nmse_db_in,
        'ssim_in': ssim_in
    })

if not loaded_data:
    print("[Error] No data loaded. Cannot generate plots.")
    exit(1)

# Save the combined comparative results to a .mat file
mat_out_path = os.path.join(output_dir, 'overall_synthesized_comparison.mat')
export_dict = {}
for d in loaded_data:
    # Use clean label name for MAT fields (replacing '+' with '_')
    clean_label = d['label'].replace('+', '_')
    export_dict[f"{clean_label}_snr"] = d['snr']
    export_dict[f"{clean_label}_mmse"] = d['mmse']
    export_dict[f"{clean_label}_nmse_db"] = d['nmse_db']
    export_dict[f"{clean_label}_ssim"] = d['ssim']
    export_dict[f"{clean_label}_mmse_in"] = d['mmse_in']
    export_dict[f"{clean_label}_nmse_db_in"] = d['nmse_db_in']
    export_dict[f"{clean_label}_ssim_in"] = d['ssim_in']

scipy.io.savemat(mat_out_path, export_dict)
print(f"Saved combined comparison MAT file to: {mat_out_path}")

# Set custom styling parameters
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 1.2

# =========================================================================
# PLOT 1: MMSE COMPARISON
# =========================================================================
plt.figure(figsize=(8.5, 6.5))

# Plot model curves
for d in loaded_data:
    plt.semilogy(d['snr'], d['mmse'], marker=d['marker'], linestyle='-', 
                 color=d['color'], linewidth=2.2, markersize=8, label=d['label'])

# Plot baseline curves (only once per type of input)
plotted_baselines = set()
for d in loaded_data:
    is_ls = 'LS' in d['label']
    baseline_label = 'LS Baseline' if is_ls else 'LS+LI Baseline'
    baseline_color = '#d6604d' if is_ls else '#7f7f7f'
    baseline_style = '--^' if is_ls else '--o'
    
    if baseline_label not in plotted_baselines:
        plt.semilogy(d['snr'], d['mmse_in'], baseline_style, color=baseline_color, 
                     linewidth=1.5, markersize=6, alpha=0.8, label=baseline_label)
        plotted_baselines.add(baseline_label)

plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('MMSE (log-scale)', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Test Set MMSE Comparison', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='best', frameon=True, facecolor='white', edgecolor='#e0e0e0')
plt.tight_layout()

mmse_plot_path = os.path.join(output_dir, 'overall_mmse_comparison.pdf')
plt.savefig(mmse_plot_path, format='pdf', dpi=300)
print(f"Saved MMSE plot to: {mmse_plot_path}")
plt.close()

# =========================================================================
# PLOT 2: NMSE (dB) COMPARISON
# =========================================================================
plt.figure(figsize=(8.5, 6.5))

# Plot model curves
for d in loaded_data:
    plt.plot(d['snr'], d['nmse_db'], marker=d['marker'], linestyle='-', 
             color=d['color'], linewidth=2.2, markersize=8, label=d['label'])

# Plot baseline curves
plotted_baselines.clear()
for d in loaded_data:
    is_ls = 'LS' in d['label']
    baseline_label = 'LS Baseline' if is_ls else 'LS+LI Baseline'
    baseline_color = '#d6604d' if is_ls else '#7f7f7f'
    baseline_style = '--^' if is_ls else '--o'
    
    if baseline_label not in plotted_baselines:
        plt.plot(d['snr'], d['nmse_db_in'], baseline_style, color=baseline_color, 
                 linewidth=1.5, markersize=6, alpha=0.8, label=baseline_label)
        plotted_baselines.add(baseline_label)

plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('NMSE (dB)', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Test Set NMSE (dB) Comparison', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='best', frameon=True, facecolor='white', edgecolor='#e0e0e0')
plt.tight_layout()

nmse_plot_path = os.path.join(output_dir, 'overall_nmse_comparison.pdf')
plt.savefig(nmse_plot_path, format='pdf', dpi=300)
print(f"Saved NMSE (dB) plot to: {nmse_plot_path}")
plt.close()

# =========================================================================
# PLOT 3: SSIM COMPARISON
# =========================================================================
plt.figure(figsize=(8.5, 6.5))

# Plot model curves
for d in loaded_data:
    plt.plot(d['snr'], d['ssim'], marker=d['marker'], linestyle='-', 
             color=d['color'], linewidth=2.2, markersize=8, label=d['label'])

# Plot baseline curves
plotted_baselines.clear()
for d in loaded_data:
    is_ls = 'LS' in d['label']
    baseline_label = 'LS Baseline' if is_ls else 'LS+LI Baseline'
    baseline_color = '#d6604d' if is_ls else '#7f7f7f'
    baseline_style = '--^' if is_ls else '--o'
    
    if baseline_label not in plotted_baselines:
        plt.plot(d['snr'], d['ssim_in'], baseline_style, color=baseline_color, 
                 linewidth=1.5, markersize=6, alpha=0.8, label=baseline_label)
        plotted_baselines.add(baseline_label)

plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold', labelpad=10)
plt.ylabel('SSIM', fontsize=12, fontweight='bold', labelpad=10)
plt.title('Test Set SSIM Comparison', fontsize=14, fontweight='bold', pad=15)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend(fontsize=10, loc='best', frameon=True, facecolor='white', edgecolor='#e0e0e0')
plt.tight_layout()

ssim_plot_path = os.path.join(output_dir, 'overall_ssim_comparison.pdf')
plt.savefig(ssim_plot_path, format='pdf', dpi=300)
print(f"Saved SSIM plot to: {ssim_plot_path}")
plt.close()