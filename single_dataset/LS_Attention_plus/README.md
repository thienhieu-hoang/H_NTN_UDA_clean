# LS_Attention_plus: Advanced Channel Estimation & Refinement Suite (5G NTN)

This folder contains four specialized refinement architectures built on top of the baseline [`train_attention_LS.py`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/train_attention_LS.py) model. They are designed to:
1. **Improve channel estimation accuracy on the Source domain** by capturing deep spatial, temporal, and transform-domain priors.
2. **Naturally specialize to the Source domain**, causing a realistic domain drop when deployed on the Target domain (due to shifts in Doppler, delay profiles, or pilot patterns `p1` vs `p2`).
3. **Establish a strong empirical justification for Unsupervised Domain Adaptation (UDA)** (CORAL, JMMD, or Adversarial UDA).

---

## 1. Overview of the Four Refinement Architectures

| Architecture | Script | Documentation | Core Mechanism | Primary Domain Sensitivity | Best For |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Two-Stage Residual Error Refiner** | [`train_attention_LS_ResidualRefine.py`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/train_attention_LS_ResidualRefine.py) | [`note_ResidualRefine.md`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/note_ResidualRefine.md) | Multi-dilation CNN predicting systematic interpolation error $\Delta \mathbf{H}$ | Pilot layout changes (`p1` $\to$ `p2`) & local Doppler phase | Shifts in pilot grid patterns |
| **2. Dual-Domain 2D-DFT Refiner** | [`train_attention_LS_DualDomain.py`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/train_attention_LS_DualDomain.py) | [`note_DualDomain.md`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/note_DualDomain.md) | Unitary 2D-DFT to Delay-Doppler domain + SE Channel Attention | Delay spread ($\tau$) & Doppler shift ($f_d$) distributions | Shifts in vehicle speed & multipath delay profiles |
| **3. Multi-Scale U-Net Refiner** | [`train_attention_LS_UNetRefine.py`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/train_attention_LS_UNetRefine.py) | [`note_UNetRefine.md`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/note_UNetRefine.md) | Down-Up hierarchy with global slot-wide bottleneck ($33 \times 7 \times 128$) | Global slot-wide time-frequency covariance structure | General domain shifts; ideal for Bottleneck UDA |
| **4. Conditional GAN (cGAN) Refiner** | [`train_attention_LS_cGAN.py`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/train_attention_LS_cGAN.py) | [`note_cGAN.md`](file:///C:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/single_dataset/LS_Attention_plus/note_cGAN.md) | Conditional PatchGAN critic on $[\hat{\mathbf{H}}_{\text{coarse}}, \mathbf{H}_{\text{cand}}]$ enforcing physical multipath sharpness | Source channel propagation manifold & delay-Doppler distribution | Eliminating regression blur; domain-adversarial UDA |

---

## 2. Fast Single-Pass Inference vs. Diffusion

All four models avoid diffusion models to strictly comply with real-time 5G slot timing:
* **5G Slot Deadline**: In 5G NTN with 30 kHz subcarrier spacing, the slot duration is only **0.5 ms**.
* **Diffusion Latency ($O(T)$)**: Takes $20 \sim 150\text{ ms}$ due to 10–50 iterative denoising steps.
* **Refinement Latency ($O(1)$)**: Executes in a **single forward pass** ($\le 1.5\text{ ms}$ on GPU/ONNX runtime), satisfying communication requirements. In cGAN, the discriminator is only used during training and discarded at inference.

---

## 3. Quick Start & Training Examples

### Example 1: Run Smoke Test on All 4 Architectures
```powershell
# Smoke test Two-Stage Residual Refiner
conda run -n TF_GPU-py3_11 python train_attention_LS_ResidualRefine.py --snr 10 --test-code --save-model

# Smoke test Dual-Domain 2D-DFT Refiner
conda run -n TF_GPU-py3_11 python train_attention_LS_DualDomain.py --snr 10 --test-code --save-model

# Smoke test Multi-Scale U-Net Refiner
conda run -n TF_GPU-py3_11 python train_attention_LS_UNetRefine.py --snr 10 --test-code --save-model

# Smoke test Conditional GAN (cGAN) Refiner
conda run -n TF_GPU-py3_11 python train_attention_LS_cGAN.py --snr 10 --test-code --save-model
```

### Example 2: Full Training with Standardization
```powershell
# Two-Stage Residual Refiner with standardization
python train_attention_LS_ResidualRefine.py --snr 10 --standardize --epochs 200 --save-model

# Dual-Domain Refiner with standardization
python train_attention_LS_DualDomain.py --snr 10 --standardize --epochs 200 --save-model

# Multi-Scale U-Net Refiner with standardization
python train_attention_LS_UNetRefine.py --snr 10 --standardize --epochs 200 --save-model

# Conditional GAN (cGAN) Refiner with standardization
python train_attention_LS_cGAN.py --snr 10 --standardize --epochs 200 --adv-weight 0.005 --save-model
```

---

## 4. How to Connect to UDA Domain Adaptation

When training with UDA (CORAL, JMMD, or Adversarial alignment):
1. **Source Domain Loss**: Compute supervised task loss (MSE + SSIM) between $\hat{\mathbf{H}}_{\text{refined}}$ and $\mathbf{H}_{\text{true}}$.
2. **Target Domain Alignment**:
   * For **Residual Refiner**: Align Pilot Attention features (`layer1`) + Post-Dilation residual features (`layer2`).
   * For **Dual-Domain**: Align Pilot Attention features (`layer1`) + Delay-Doppler cluster attention features (`layer2`).
   * For **Multi-Scale U-Net**: Align Pilot Attention features (`layer1`) + U-Net Bottleneck features (`layer2`).
   * For **cGAN Refiner**: Align Pilot Attention features (`layer1`) + Generator Refiner features (`layer2`), or re-purpose the trained Discriminator with a Gradient Reversal Layer (`DANN`).
3. **Projection Head**: Route extracted features through a dedicated `Dense(128) -> LayerNorm -> GeLU -> Dense(64)` projection head before calculating covariance/MMD distance.
