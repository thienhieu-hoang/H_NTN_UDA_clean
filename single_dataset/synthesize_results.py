import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

# Directories and parameters
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(THIS_DIR, 'DnCNN_Attention_DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps_LI')
SNR_LIST = [-10, -5, 0, 5, 10, 15]

def synthesize_results(input_type):
    # Check if there are any subfolders starting with the prefix (e.g. "LI_" or "LS_")
    # excluding the synthesize output folder itself
    if os.path.exists(DATASET_DIR):
        subdirs = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
        matching_subdirs = [d for d in subdirs if d.lower().startswith(f"{input_type.lower()}_") and not d.lower().endswith("_synthesize")]
        if not matching_subdirs:
            print(f"\n=== Skipping synthesis for input_type: {input_type} (No matching {input_type}_ subfolders found) ===")
            return
            
    print(f"\n=== Synthesizing results for input_type: {input_type} ===")
    
    # Initialize dictionary to collect lists
    collected = {
        'snr': [],
        # Train
        'mmse_train': [], 'nmse_train': [], 'nmse_train_db': [], 'ssim_train': [],
        'mmse_input_train': [], 'nmse_input_train': [], 'nmse_input_train_db': [], 'ssim_input_train': [],
        # Val
        'mmse_val': [], 'nmse_val': [], 'nmse_val_db': [], 'ssim_val': [],
        'mmse_input_val': [], 'nmse_input_val': [], 'nmse_input_val_db': [], 'ssim_input_val': [],
        # Test
        'mmse_test': [], 'nmse_test': [], 'nmse_test_db': [], 'ssim_test': [],
        'mmse_input_test': [], 'nmse_input_test': [], 'nmse_input_test_db': [], 'ssim_input_test': []
    }
    
    for snr in SNR_LIST:
        folder_name = f"{input_type}_{snr}"
        mat_path = os.path.join(DATASET_DIR, folder_name, 'results', 'evaluation_results.mat')
        
        if not os.path.exists(mat_path):
            print(f"Warning: File not found: {mat_path}")
            continue
            
        mat = scipy.io.loadmat(mat_path)
        collected['snr'].append(snr)
        
        for key in collected.keys():
            if key == 'snr':
                continue
            if key in mat:
                # Extract scalar value from mat array
                val = float(np.squeeze(mat[key]))
                collected[key].append(val)
            else:
                print(f"Warning: Key {key} not found in {mat_path}")
                collected[key].append(np.nan)
                
    # Convert lists to numpy arrays
    for key in collected.keys():
        collected[key] = np.array(collected[key])
        
    # Create output directory
    output_dir = os.path.join(DATASET_DIR, f"{input_type}_synthesize")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to .mat
    mat_output_path = os.path.join(output_dir, 'synthesized_results.mat')
    scipy.io.savemat(mat_output_path, collected)
    print(f"Saved synthesized .mat file to: {mat_output_path}")
    
    # Plot figures
    snrs = collected['snr']
    # 1. MMSE Figure
    plt.figure(figsize=(8, 6))
    plt.semilogy(snrs, collected['mmse_input_train'], 'o--', label='Input (Train)', color='gray')
    plt.semilogy(snrs, collected['mmse_train'], 'o-', label='CNN Output (Train)', color='blue')
    plt.semilogy(snrs, collected['mmse_input_val'], 's--', label='Input (Val)', color='lightcoral')
    plt.semilogy(snrs, collected['mmse_val'], 's-', label='CNN Output (Val)', color='red')
    plt.semilogy(snrs, collected['mmse_input_test'], '^--', label='Input (Test)', color='lightgreen')
    plt.semilogy(snrs, collected['mmse_test'], '^-', label='CNN Output (Test)', color='green')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('MMSE', fontsize=12)
    plt.title(f'MMSE Comparison over SNR ({input_type.upper()})', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    mmse_plot_path = os.path.join(output_dir, 'mmse_comparison.pdf')
    plt.savefig(mmse_plot_path, format='pdf', dpi=300)
    plt.close()
    
    # 2. NMSE (dB) Figure
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, collected['nmse_input_train_db'], 'o--', label='Input (Train)', color='gray')
    plt.plot(snrs, collected['nmse_train_db'], 'o-', label='CNN Output (Train)', color='blue')
    plt.plot(snrs, collected['nmse_input_val_db'], 's--', label='Input (Val)', color='lightcoral')
    plt.plot(snrs, collected['nmse_val_db'], 's-', label='CNN Output (Val)', color='red')
    plt.plot(snrs, collected['nmse_input_test_db'], '^--', label='Input (Test)', color='lightgreen')
    plt.plot(snrs, collected['nmse_test_db'], '^-', label='CNN Output (Test)', color='green')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('NMSE (dB)', fontsize=12)
    plt.title(f'NMSE (dB) Comparison over SNR ({input_type.upper()})', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    nmse_plot_path = os.path.join(output_dir, 'nmse_comparison.pdf')
    plt.savefig(nmse_plot_path, format='pdf', dpi=300)
    plt.close()
    
    # 3. SSIM Figure
    plt.figure(figsize=(8, 6))
    plt.plot(snrs, collected['ssim_input_train'], 'o--', label='Input (Train)', color='gray')
    plt.plot(snrs, collected['ssim_train'], 'o-', label='CNN Output (Train)', color='blue')
    plt.plot(snrs, collected['ssim_input_val'], 's--', label='Input (Val)', color='lightcoral')
    plt.plot(snrs, collected['ssim_val'], 's-', label='CNN Output (Val)', color='red')
    plt.plot(snrs, collected['ssim_input_test'], '^--', label='Input (Test)', color='lightgreen')
    plt.plot(snrs, collected['ssim_test'], '^-', label='CNN Output (Test)', color='green')
    plt.xlabel('SNR (dB)', fontsize=12)
    plt.ylabel('SSIM', fontsize=12)
    plt.title(f'SSIM Comparison over SNR ({input_type.upper()})', fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    ssim_plot_path = os.path.join(output_dir, 'ssim_comparison.pdf')
    plt.savefig(ssim_plot_path, format='pdf', dpi=300)
    plt.close()
    
    print(f"Plots saved to: {output_dir}")

if __name__ == '__main__':
    synthesize_results('LI')
    synthesize_results('LS')
