"""
Approach C: Physics-Guided Attention Map CORAL Training Script
Trains / fine-tunes LS_Attention model via joint supervised loss on source domain
and Physics-Warped Multi-Head Attention Alignment (L_physics_attn) on target domain.
"""

import os
import sys
import argparse
import time
import numpy as np

# Add project root and CORAL directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
coral_dir = os.path.join(project_root, 'CORAL')
for p in [project_root, coral_dir, os.path.join(project_root, 'math_translation')]:
    if p not in sys.path:
        sys.path.insert(0, p)

from math_translation.core.estimate_pilot_shifts import estimate_delay_dilation_autocorr
from math_translation.core.io_utils import load_channel_mat
from math_translation.approachC_physics_coral.physics_coral_loss import tf_physics_attn_map_coral_loss


def run_physics_coral_training(
    source_dir,
    target_dir,
    save_dir,
    snr=10,
    beta_tau=None,
    epochs=100,
    batch_size=16,
    domain_weight=1.0,
    norm_mode="rms"
):
    print("\n" + "=" * 80)
    print("  Approach C: Physics-Guided Attention CORAL UDA Training")
    print("=" * 80)
    print(f"Source Directory : {source_dir}")
    print(f"Target Directory : {target_dir}")
    print(f"Save Directory   : {save_dir}")
    print(f"Channel SNR      : {snr} dB")
    print(f"Epochs           : {epochs}")
    print(f"Batch Size       : {batch_size}")
    print(f"CORAL Weight     : {domain_weight}")
    print(f"Normalization    : {norm_mode.upper()}")
    print("=" * 80 + "\n")

    os.makedirs(save_dir, exist_ok=True)

    # 1. Blind parameter estimation if beta_tau not provided
    if beta_tau is None or beta_tau <= 0:
        print("[*] Performing blind Delay Dilation estimation from pilots...")
        try:
            _, p_src, r_p, c_p, _ = load_channel_mat(source_dir, 'ls')
            _, p_tgt, _, _, _ = load_channel_mat(target_dir, 'ls')
            b_est, s_info, t_info = estimate_delay_dilation_autocorr(p_src, p_tgt, r_p, c_p)
            beta_tau = b_est
            print(f"  [Estimated] Beta_tau: {beta_tau:.3f} (Src lag: {s_info[2]:.1f}, Tgt lag: {t_info[2]:.1f})")
        except Exception as e:
            beta_tau = 3.0
            print(f"  [Fallback] Could not estimate automatically ({e}). Using physical default beta_tau = 3.0")
    else:
        print(f"  [Specified] Beta_tau: {beta_tau:.3f}")

    # Build execution command forwarding to run_LS_Attention_AttnMapCORAL.py with physics loss
    script_path = os.path.join(coral_dir, "run_LS_Attention_AttnMapCORAL.py")
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Underlying UDA script not found at: {script_path}")

    # Normalize flag
    norm_flag = f"--{norm_mode.lower()}" if norm_mode.lower() in ["rms", "standardize"] else ""

    cmd = [
        sys.executable,
        script_path,
        "--source-dir", source_dir,
        "--target-dir", target_dir,
        "--save-dir", save_dir,
        "--snr", str(snr),
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
        "--domain-weight", str(domain_weight)
    ]
    if norm_flag:
        cmd.append(norm_flag)

    print(f"\n[*] Executing Physics-Guided Attention CORAL training...")
    print(f"    Command: {' '.join(cmd)}\n")

    import subprocess
    ret = subprocess.run(cmd)

    # Save metadata note
    note_file = os.path.join(save_dir, "physics_coral_config.md")
    with open(note_file, "w") as fh:
        fh.write("# Physics-Guided Attention CORAL Training Run\n\n")
        fh.write(f"- **Method:** Approach C (Physics-Warped Attention Alignment)\n")
        fh.write(f"- **Source Domain:** `{source_dir}`\n")
        fh.write(f"- **Target Domain:** `{target_dir}`\n")
        fh.write(f"- **Beta_tau:** `{beta_tau:.3f}`\n")
        fh.write(f"- **CORAL Domain Weight:** `{domain_weight}`\n")
        fh.write(f"- **Epochs:** `{epochs}`\n")
        fh.write(f"- **Normalization:** `{norm_mode}`\n")

    return ret.returncode


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Approach C: Physics-Guided Attention CORAL Training")
    parser.add_argument("--source-dir", type=str, default=r"generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz")
    parser.add_argument("--target-dir", type=str, default=r"generatedChan\OpenNTN\DUR300nsFix_NLoS_port1_Apos2_2p18G_600km_30deg_r15km_20to30mps")
    parser.add_argument("--save-dir", type=str, default=r"math_translation\results\approachC_physics_coral")
    parser.add_argument("--snr", type=int, default=10)
    parser.add_argument("--beta-tau", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--domain-weight", type=float, default=1.0)
    parser.add_argument("--norm-mode", type=str, default="rms", choices=["rms", "standardize", "minmax"])

    args = parser.parse_args()
    code = run_physics_coral_training(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        save_dir=args.save_dir,
        snr=args.snr,
        beta_tau=args.beta_tau,
        epochs=args.epochs,
        batch_size=args.batch_size,
        domain_weight=args.domain_weight,
        norm_mode=args.norm_mode
    )
    sys.exit(code)

