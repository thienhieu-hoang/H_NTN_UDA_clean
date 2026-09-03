"""
====================================================================================================
JMMD cGAN Domain Adaptation for NTN Channel Estimation (OpenNTN & MATLAB) - High-Performance
====================================================================================================

Overview
--------
This script trains and evaluates a Conditional Generative Adversarial Network (cGAN) based on 
Pix2Pix (UNet Generator + PatchGAN Discriminator with WGAN-GP) combined with Joint Maximum Mean 
Discrepancy (JMMD) Unsupervised Domain Adaptation (UDA) for 5G Non-Terrestrial Network (NTN) 
channel estimation.

Domain shift (e.g., between different user velocities, propagation delays, or TDL channel profiles)
is mitigated using JMMD multi-scale Gaussian RBF Reproducing Kernel Hilbert Space (RKHS) loss, 
which aligns the joint distributions of multi-layer intermediate generator features between the 
source and target domains.

Performance Optimizations
-------------------------
- Pure `@tf.function` Graph Execution: Compiled CUDA kernels execute WGAN-GP, JMMD, and Generator
  steps with zero Python eager-mode dispatch overhead.
- In-Memory GPU Batching: Zero per-batch CPU array copies or repeated transposition.
- Fast Vectorized Validation & SSIM: Skips redundant discriminator passes during validation.

Dataset Splitting (3-Way Split)
--------------------------------
- Train Set (Default 70%): Labeled source domain + Unlabeled target domain for JMMD adaptation.
- Validation Set (Default 15%): Periodic evaluation and model checkpointing during training.
- Test Set (Default 15%): Final held-out evaluation exported to `testChannel_source.mat` and 
  `testChannel_target.mat` for standalone MATLAB BER simulations.

Adaptive Dataset Directory Input Parser (`--source-dir` & `--target-dir`):
-------------------------------------------------------------------------
The input parser supports 3 flexible input formats without requiring you to specify whether a dataset 
is from MATLAB or OpenNTN:

1. Scenario Folder Name / Substring (Auto-Discovered):
   You can pass just the folder name or scenario substring. The loader automatically scans both 
   `generatedChan/MATLAB/` and `generatedChan/OpenNTN/` to find the matching directory:
     --source-dir A100_2p18e9_600km_70deg_30kHz
     --target-dir DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps

2. Relative Path from Project Root:
   You can pass the relative path:
     --source-dir generatedChan/MATLAB/A100_2p18e9_600km_70deg_30kHz
     --target-dir generatedChan/OpenNTN/DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps

3. Full Absolute Path:
   You can pass the full path on disk:
     --source-dir "C:/Users/.../generatedChan/MATLAB/A100_2p18e9_600km_70deg_30kHz"
     --target-dir "C:/Users/.../generatedChan/OpenNTN/DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps"

How the Loader Resolves Files:
- Automatically maps requested `--snr` (e.g. 5) to matching subfolder (`SNR_5dB`, `5dB`, `SNR_5`, `5`, etc.).
- Automatically detects the dataset `.mat` file (`matlabNTN.mat` or `channel_dur_randomizedUE.mat`).
- Supports both MATLAB v7 (scipy.io) and v7.3 HDF5 (h5py) file structures seamlessly.

Usage Examples
--------------
    # Run full JMMD cGAN training on GPU at SNR = 5 dB with MATLAB A100 vs OpenNTN
    python run_JMMD_cGAN.py --source-dir A100_2p18e9_600km_70deg_30kHz --target-dir DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps --snr 5 --jmmd-layers d4 --domain-weight 0.5 --save-features

    # Run multi-layer JMMD (d3 + d4) cGAN
    python run_JMMD_cGAN.py --snr 5 --jmmd-layers d3 d4 --domain-weight 0.5

    # Run Source-Only Baseline (no JMMD domain adaptation)
    python run_JMMD_cGAN.py --snr 5 --only-source

    # Train with sample-wise zero-mean unit-variance standardization
    python run_JMMD_cGAN.py --snr 5 --standardize

    # Quick sanity check (runs in < 15 seconds)
    python run_JMMD_cGAN.py --test-code --snr 5
====================================================================================================
"""

# NumPy compatibility patch for NumPy 2.x
import numpy as np
if not hasattr(np, 'complex_'):
    np.complex_ = np.complex128
if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'string_'):
    np.string_ = np.bytes_
if not hasattr(np, 'unicode_'):
    np.unicode_ = np.str_

try:
    if hasattr(np, 'sctypeDict'):
        if 'string_' not in np.sctypeDict:
            np.sctypeDict['string_'] = np.bytes_
        if 'unicode_' not in np.sctypeDict:
            np.sctypeDict['unicode_'] = np.str_
    if hasattr(np, 'typeDict'):
        if 'string_' not in np.typeDict:
            np.typeDict['string_'] = np.bytes_
        if 'unicode_' not in np.typeDict:
            np.typeDict['unicode_'] = np.str_
except Exception:
    pass

import os
import sys
import time
import argparse
import scipy
from scipy.io import savemat, loadmat
import h5py
import tensorflow as tf
from tensorflow.image import ssim as tf_ssim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
DEFAULT_SOURCE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"
DEFAULT_TARGET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps"
DEFAULT_SAVE_DIR = ""
DEFAULT_SNR = 5
DEFAULT_TYPE = "LS"
DEFAULT_ONLY_SOURCE = False
DEFAULT_JMMD_LAYERS = ["d4"]
DEFAULT_JMMD_MODE = "joint"
DEFAULT_KERNEL_NUM = 5
DEFAULT_KERNEL_MUL = 2.0
DEFAULT_SAVE_FEATURES = False
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
DEFAULT_N_EPOCHS = 300
DEFAULT_BATCH_SIZE = 16
DEFAULT_ADV_WEIGHT = 0.005
DEFAULT_EST_WEIGHT = 1.0
DEFAULT_DOMAIN_WEIGHT = 0.5
DEFAULT_TEMPORAL_WEIGHT = 0.02
DEFAULT_FREQUENCY_WEIGHT = 0.1
DEFAULT_SSIM_WEIGHT = 0.1
DEFAULT_GP_WEIGHT = 10.0
DEFAULT_LR_G = 2e-4
DEFAULT_LR_D = 2e-4
DEFAULT_DISC_STEPS = 1
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))


# =============================================================================
# 1. GENERATOR & DISCRIMINATOR NETWORK MODULES
# =============================================================================
class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer (epsilon = 1e-5)."""
    def __init__(self, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale', shape=(input_shape[-1],), initializer='ones', trainable=True
        )
        self.offset = self.add_weight(
            name='offset', shape=(input_shape[-1],), initializer='zeros', trainable=True
        )

    def call(self, x, training=False):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class UNetBlock(tf.keras.layers.Layer):
    """Encoder DownBlock for UNet Generator."""
    def __init__(self, filters, apply_dropout=False, kernel_size=(4, 3), strides=(2, 1), gen_l2=None):
        super().__init__()
        kernel_reg = tf.keras.regularizers.l2(gen_l2) if gen_l2 else None
        self.conv = tf.keras.layers.Conv2D(
            filters, kernel_size=kernel_size, strides=strides, padding='valid', kernel_regularizer=kernel_reg
        )
        self.norm = InstanceNormalization()
        self.dropout = tf.keras.layers.Dropout(0.3) if apply_dropout else None

    def call(self, x, training=False):
        x = self.conv(x)
        x = self.norm(x, training=training)
        x = tf.nn.leaky_relu(x, alpha=0.2)
        if self.dropout:
            x = self.dropout(x, training=training)
        return x


class UNetUpBlock(tf.keras.layers.Layer):
    """Decoder UpBlock for UNet Generator with Skip Connection."""
    def __init__(self, filters, apply_dropout=False, kernel_size=(4, 3), strides=(2, 1), gen_l2=None, dropOut_rate=0.3):
        super().__init__()
        kernel_reg = tf.keras.regularizers.l2(gen_l2) if gen_l2 else None
        self.deconv = tf.keras.layers.Conv2DTranspose(
            filters, kernel_size=kernel_size, strides=strides, padding='valid', kernel_regularizer=kernel_reg
        )
        self.norm = InstanceNormalization()
        self.dropout = tf.keras.layers.Dropout(dropOut_rate) if apply_dropout else None

    def call(self, x, skip, training=False):
        x = self.deconv(x)
        # Crop if symbol dimension expands past 14
        if tf.shape(x)[2] > 14:
            x = x[:, :, 1:15, :]
        x = self.norm(x, training=training)
        x = tf.nn.relu(x)
        if self.dropout:
            x = self.dropout(x, training=training)
        return tf.concat([x, skip], axis=-1)


class Pix2PixGenerator(tf.keras.Model):
    """UNet Generator for 5G NTN Channel Estimation."""
    def __init__(self, output_channels=2, gen_l2=None,
                 dropOut_layers=['u1', 'u2'], dropOut_rate=0.3, extract_layers=['d4']):
        super().__init__()
        kernel_reg = tf.keras.regularizers.l2(gen_l2) if gen_l2 else None
        self.extract_layers = [lyr.lower() for lyr in extract_layers]

        # Encoder (Downsampling along subcarrier axis: 132 -> 65 -> 32 -> 15 -> 7)
        self.down1 = UNetBlock(32, apply_dropout=False, kernel_size=(4, 3), strides=(2, 1), gen_l2=gen_l2)
        self.down2 = UNetBlock(64, kernel_size=(3, 3), strides=(2, 1), gen_l2=gen_l2)
        self.down3 = UNetBlock(128, kernel_size=(4, 3), strides=(2, 1), gen_l2=gen_l2)
        self.down4 = UNetBlock(256, kernel_size=(3, 3), strides=(2, 1), gen_l2=gen_l2)

        # Decoder (Upsampling with skip connections: 7 -> 15 -> 32 -> 65 -> 132)
        self.up1 = UNetUpBlock(128, apply_dropout='u1' in dropOut_layers, kernel_size=(3, 3), strides=(2, 1), gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        self.up2 = UNetUpBlock(64, apply_dropout='u2' in dropOut_layers, kernel_size=(4, 3), strides=(2, 1), gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        self.up3 = UNetUpBlock(32, apply_dropout='u3' in dropOut_layers, kernel_size=(3, 3), strides=(2, 1), gen_l2=gen_l2, dropOut_rate=dropOut_rate)
        self.last = tf.keras.layers.Conv2DTranspose(
            output_channels, kernel_size=(4, 3), strides=(2, 1), padding='valid', kernel_regularizer=kernel_reg
        )

    def call(self, x, training=False, return_features=False):
        d1 = self.down1(x, training=training)      # [B, 65, 14, 32]
        d2 = self.down2(d1, training=training)     # [B, 32, 14, 64]
        d3 = self.down3(d2, training=training)     # [B, 15, 14, 128]
        d4 = self.down4(d3, training=training)     # [B,  7, 14, 256]

        u1 = self.up1(d4, d3, training=training)   # [B, 15, 14, 256]
        u2 = self.up2(u1, d2, training=training)   # [B, 32, 14, 128]
        u3 = self.up3(u2, d1, training=training)   # [B, 65, 14, 64]
        u4 = self.last(u3)                          # [B, 132, 14, 2]
        if tf.shape(u4)[2] > 14:
            u4 = u4[:, :, 1:15, :]

        features = []
        layer_map = {'d1': d1, 'd2': d2, 'd3': d3, 'd4': d4, 'u1': u1, 'u2': u2, 'u3': u3}
        for layer_name in self.extract_layers:
            if layer_name in layer_map:
                features.append(layer_map[layer_name])

        return u4, features


class PatchGANDiscriminator(tf.keras.Model):
    """PatchGAN Discriminator for WGAN-GP Adversarial Training."""
    def __init__(self, filters=[32, 64, 128, 256], disc_l2=1e-5):
        super().__init__()
        kernel_reg = tf.keras.regularizers.l2(disc_l2) if disc_l2 else None
        self.conv1 = tf.keras.layers.Conv2D(filters[0], kernel_size=(4, 3), strides=(2, 1), padding='valid', kernel_regularizer=kernel_reg)
        self.conv2 = tf.keras.layers.Conv2D(filters[1], kernel_size=(3, 3), strides=(2, 1), padding='valid', kernel_regularizer=kernel_reg)
        self.norm2 = InstanceNormalization()
        self.conv3 = tf.keras.layers.Conv2D(filters[2], kernel_size=(4, 3), strides=(2, 1), padding='valid', kernel_regularizer=kernel_reg)
        self.norm3 = InstanceNormalization()
        self.conv4 = tf.keras.layers.Conv2D(filters[3], kernel_size=(3, 3), strides=(2, 1), padding='valid', kernel_regularizer=kernel_reg)
        self.norm4 = InstanceNormalization()
        self.last = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), strides=(2, 1), padding='valid', kernel_regularizer=kernel_reg)

    def call(self, x, training=False):
        x = tf.nn.leaky_relu(self.conv1(x), alpha=0.2)
        x = tf.nn.leaky_relu(self.norm2(self.conv2(x), training=training), alpha=0.2)
        x = tf.nn.leaky_relu(self.norm3(self.conv3(x), training=training), alpha=0.2)
        x = tf.nn.leaky_relu(self.norm4(self.conv4(x), training=training), alpha=0.2)
        return self.last(x)


# =============================================================================
# 2. FAST VECTORIZED NORMALIZATION & LOSS HELPERS
# =============================================================================
@tf.function
def batch_minmax_scale(x: tf.Tensor, y: tf.Tensor, lower_range: float = -1.0):
    """Vectorized per-sample MinMax scaling directly on GPU tensors [B, 132, 14, 2]."""
    x_min = tf.reduce_min(x, axis=[1, 2], keepdims=True)
    x_max = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    scale = tf.clip_by_value(x_max - x_min, 1e-12, tf.float32.max)

    x_scaled = (x - x_min) / scale
    y_scaled = (y - x_min) / scale

    if lower_range == -1.0:
        x_scaled = x_scaled * 2.0 - 1.0
        y_scaled = y_scaled * 2.0 - 1.0

    return x_scaled, y_scaled, x_min, scale


@tf.function
def batch_standardize(x: tf.Tensor, y: tf.Tensor):
    """Vectorized per-sample zero-mean unit-variance standardization directly on GPU tensors [B, 132, 14, 2]."""
    x_mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
    x_mean_sq = tf.reduce_mean(tf.square(x), axis=[1, 2], keepdims=True)
    x_var = x_mean_sq - tf.square(x_mean)
    x_std = tf.sqrt(tf.clip_by_value(x_var, 1e-12, tf.float32.max))

    x_scaled = (x - x_mean) / x_std
    y_scaled = (y - x_mean) / x_std

    return x_scaled, y_scaled, x_mean, x_std


@tf.function
def compute_gradient_penalty(discriminator, real_samples, fake_samples):
    """Vectorized WGAN-GP Gradient Penalty computation."""
    batch_size = tf.shape(real_samples)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=real_samples.dtype)
    interpolated = alpha * real_samples + (1.0 - alpha) * fake_samples

    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)

    grads = gp_tape.gradient(pred, [interpolated])[0]
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
    gp = tf.reduce_mean(tf.square(grad_norm - 1.0))
    return gp


@tf.function
def compute_gaussian_kernel_matrix(x1: tf.Tensor, x2: tf.Tensor, kernel_mul: float = 2.0, kernel_num: int = 5) -> tf.Tensor:
    """
    Compute Multi-Scale Gaussian RBF Kernel Matrix K(x1, x2) of shape [B1, B2].
    k(x_i, x_j) = sum_{m=1}^M exp( - ||x_i - x_j||^2 / bw_m )
    """
    # Global spatial pooling if 4D conv feature map
    if len(x1.shape) == 4:
        x1_flat = tf.reduce_mean(x1, axis=[1, 2])
        x2_flat = tf.reduce_mean(x2, axis=[1, 2])
    else:
        x1_flat = tf.reshape(x1, [tf.shape(x1)[0], -1])
        x2_flat = tf.reshape(x2, [tf.shape(x2)[0], -1])
    
    # L2 normalize for robust RKHS geometry
    x1_norm = tf.nn.l2_normalize(x1_flat, axis=-1)
    x2_norm = tf.nn.l2_normalize(x2_flat, axis=-1)
    
    r1 = tf.reduce_sum(tf.square(x1_norm), axis=-1, keepdims=True)  # [B1, 1]
    r2 = tf.reduce_sum(tf.square(x2_norm), axis=-1, keepdims=True)  # [B2, 1]
    dist_sq = tf.maximum(r1 - 2.0 * tf.matmul(x1_norm, x2_norm, transpose_b=True) + tf.transpose(r2), 0.0)
    
    bandwidth = tf.reduce_mean(dist_sq) + 1e-12
    bandwidth_base = bandwidth / (kernel_mul ** (kernel_num // 2))
    kernel_val = tf.zeros_like(dist_sq)
    for i in range(kernel_num):
        bw_i = bandwidth_base * (kernel_mul ** float(i))
        kernel_val = kernel_val + tf.exp(-dist_sq / tf.maximum(bw_i, 1e-12))
    
    return kernel_val


@tf.function
def compute_jmmd_loss(source_features: list, target_features: list,
                      kernel_mul: float = 2.0, kernel_num: int = 5, mode: str = "joint") -> tf.Tensor:
    """
    Joint Maximum Mean Discrepancy (JMMD) Loss across generator representations.
    """
    if len(source_features) == 0:
        return tf.constant(0.0, dtype=tf.float32)
    
    B_s = tf.shape(source_features[0])[0]
    B_t = tf.shape(target_features[0])[0]
    
    if mode == "joint":
        K_ss = tf.ones([B_s, B_s], dtype=tf.float32)
        K_tt = tf.ones([B_t, B_t], dtype=tf.float32)
        K_st = tf.ones([B_s, B_t], dtype=tf.float32)
        
        for f_s, f_t in zip(source_features, target_features):
            K_ss = K_ss * compute_gaussian_kernel_matrix(f_s, f_s, kernel_mul, kernel_num)
            K_tt = K_tt * compute_gaussian_kernel_matrix(f_t, f_t, kernel_mul, kernel_num)
            K_st = K_st * compute_gaussian_kernel_matrix(f_s, f_t, kernel_mul, kernel_num)
        
        loss = tf.reduce_mean(K_ss) + tf.reduce_mean(K_tt) - 2.0 * tf.reduce_mean(K_st)
        return tf.maximum(loss, 0.0)
    else:
        layer_losses = []
        for f_s, f_t in zip(source_features, target_features):
            k_ss = compute_gaussian_kernel_matrix(f_s, f_s, kernel_mul, kernel_num)
            k_tt = compute_gaussian_kernel_matrix(f_t, f_t, kernel_mul, kernel_num)
            k_st = compute_gaussian_kernel_matrix(f_s, f_t, kernel_mul, kernel_num)
            l_mmd = tf.reduce_mean(k_ss) + tf.reduce_mean(k_tt) - 2.0 * tf.reduce_mean(k_st)
            layer_losses.append(tf.maximum(l_mmd, 0.0))
        return tf.reduce_mean(layer_losses)


@tf.function
def compute_smoothness_loss(pred_grid: tf.Tensor):
    """Temporal and Frequency TV Smoothness Loss on predicted channel grid."""
    time_diff = pred_grid[:, :, 1:, :] - pred_grid[:, :, :-1, :]
    freq_diff = pred_grid[:, 1:, :, :] - pred_grid[:, :-1, :, :]
    loss_temporal = tf.reduce_mean(tf.square(time_diff))
    loss_frequency = tf.reduce_mean(tf.square(freq_diff))
    return loss_temporal, loss_frequency


# =============================================================================
# 3. COMPILED GPU TRAINING STEPS
# =============================================================================
@tf.function
def train_step_jmmd(
    generator, discriminator, gen_opt, disc_opt,
    src_input, src_real, tgt_input,
    adv_w, est_w, dom_w, temp_w, freq_w, ssim_w, gp_w,
    kernel_mul=2.0, kernel_num=5, jmmd_mode="joint", ssim_max_val=2.0
):
    """Compiled high-performance GPU step for JMMD cGAN UDA."""
    # -------------------------------------------------------------
    # 1. Discriminator Step (WGAN-GP)
    # -------------------------------------------------------------
    with tf.GradientTape() as disc_tape:
        src_fake, _ = generator(src_input, training=True)
        disc_real = discriminator(src_real, training=True)
        disc_fake = discriminator(src_fake, training=True)

        d_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gp = compute_gradient_penalty(discriminator, src_real, src_fake)
        total_d_loss = d_loss + gp_w * gp

    d_grads = disc_tape.gradient(total_d_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # -------------------------------------------------------------
    # 2. Generator Step (Adversarial + Estimation + Smoothness + JMMD)
    # -------------------------------------------------------------
    with tf.GradientTape() as gen_tape:
        src_fake, src_feats = generator(src_input, training=True, return_features=True)
        _, tgt_feats = generator(tgt_input, training=True, return_features=True)

        disc_fake_g = discriminator(src_fake, training=True)
        g_adv_loss = -tf.reduce_mean(disc_fake_g)

        l1_est = tf.reduce_mean(tf.abs(src_real - src_fake))
        ssim_loss = 1.0 - tf.reduce_mean(tf_ssim(src_real, src_fake, max_val=ssim_max_val))
        g_est_loss = l1_est + ssim_w * ssim_loss

        l_temp, l_freq = compute_smoothness_loss(src_fake)
        g_smooth_loss = temp_w * l_temp + freq_w * l_freq

        jmmd_loss = compute_jmmd_loss(src_feats, tgt_feats, kernel_mul, kernel_num, jmmd_mode)

        total_g_loss = (
            adv_w * g_adv_loss +
            est_w * g_est_loss +
            g_smooth_loss +
            dom_w * jmmd_loss
        )

    g_grads = gen_tape.gradient(total_g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return total_g_loss, total_d_loss, g_est_loss, jmmd_loss, g_adv_loss


@tf.function
def train_step_source_only(
    generator, discriminator, gen_opt, disc_opt,
    src_input, src_real,
    adv_w, est_w, temp_w, freq_w, ssim_w, gp_w, ssim_max_val=2.0
):
    """Source-only WGAN-GP training step (no JMMD adaptation)."""
    with tf.GradientTape() as disc_tape:
        src_fake, _ = generator(src_input, training=True)
        disc_real = discriminator(src_real, training=True)
        disc_fake = discriminator(src_fake, training=True)

        d_loss = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
        gp = compute_gradient_penalty(discriminator, src_real, src_fake)
        total_d_loss = d_loss + gp_w * gp

    d_grads = disc_tape.gradient(total_d_loss, discriminator.trainable_variables)
    disc_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    with tf.GradientTape() as gen_tape:
        src_fake, _ = generator(src_input, training=True)
        disc_fake_g = discriminator(src_fake, training=True)
        g_adv_loss = -tf.reduce_mean(disc_fake_g)

        l1_est = tf.reduce_mean(tf.abs(src_real - src_fake))
        ssim_loss = 1.0 - tf.reduce_mean(tf_ssim(src_real, src_fake, max_val=ssim_max_val))
        g_est_loss = l1_est + ssim_w * ssim_loss

        l_temp, l_freq = compute_smoothness_loss(src_fake)
        g_smooth_loss = temp_w * l_temp + freq_w * l_freq

        total_g_loss = adv_w * g_adv_loss + est_w * g_est_loss + g_smooth_loss

    g_grads = gen_tape.gradient(total_g_loss, generator.trainable_variables)
    gen_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return total_g_loss, total_d_loss, g_est_loss, tf.constant(0.0, dtype=tf.float32), g_adv_loss


@tf.function
def val_step_fast(generator, val_input, val_real):
    """Fast validation pass executing purely on GPU (L1 + SSIM)."""
    val_fake, _ = generator(val_input, training=False)
    l1_est = tf.reduce_mean(tf.abs(val_real - val_fake))
    ssim_val = tf.reduce_mean(tf_ssim(val_real, val_fake, max_val=2.0))
    return l1_est, ssim_val


# =============================================================================
# 4. DATASET RESOLUTION & EXTRACTION HELPERS
# =============================================================================
def find_any_mat_file(base_dir: str) -> str:
    """Recursively search base_dir for the first valid channel .mat file."""
    if not os.path.exists(base_dir):
        return None
    for root, _, files in os.walk(base_dir):
        for f in sorted(files):
            if f.endswith('.mat') and not f.startswith(('inferredChannel', 'testChannel', 'training_history', 'extracted_features', 'synthesized_results')):
                return os.path.join(root, f)
    return None


def get_mat_file(data_root: str, snr: int = 5) -> str:
    """
    Robustly locate the .mat data file for the requested SNR, supporting:
    - SNR folder variations: 'SNR_-10dB', '-10dB', 'SNR_-10', '-10', '5dB', etc.
    - Relative paths from workspace root or script directory
    - Scenario name substring matching (e.g. 'A100_2p18e9...' matching in generatedChan/MATLAB or generatedChan/OpenNTN)
    """
    if os.path.isfile(data_root) and data_root.endswith('.mat'):
        return os.path.abspath(data_root)

    candidate_roots = []
    if data_root:
        if os.path.isabs(data_root):
            candidate_roots.append(data_root)
        else:
            candidate_roots.append(os.path.abspath(data_root))
            candidate_roots.append(os.path.join(project_root, data_root))
            candidate_roots.append(os.path.join(current_dir, data_root))
            
            # Scenario substring search in generatedChan/
            for parent in [os.path.join(project_root, 'generatedChan', 'MATLAB'),
                           os.path.join(project_root, 'generatedChan', 'OpenNTN')]:
                if os.path.isdir(parent):
                    base_name = os.path.basename(data_root.rstrip('\\/'))
                    for entry in os.listdir(parent):
                        if base_name.lower() in entry.lower():
                            candidate_roots.append(os.path.join(parent, entry))

    # Add default fallbacks
    candidate_roots.extend([
        os.path.join(project_root, 'generatedChan', 'OpenNTN', 'DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps'),
        os.path.join(project_root, 'generatedChan', 'OpenNTN', 'DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps'),
        os.path.join(project_root, 'generatedChan', 'MATLAB', 'A100_2p18e9_600km_70deg_30kHz')
    ])

    snr_variations = [
        f"SNR_{snr}dB",
        f"{snr}dB",
        f"SNR_{snr}",
        f"{snr}",
        f"SNR_{snr:02d}dB",
        f"snr_{snr}db"
    ]

    for root in candidate_roots:
        if not os.path.isdir(root):
            continue
        for snr_var in snr_variations:
            snr_dir = os.path.join(root, snr_var)
            if os.path.isdir(snr_dir):
                mat_file = find_any_mat_file(snr_dir)
                if mat_file:
                    return os.path.abspath(mat_file)
        mat_file = find_any_mat_file(root)
        if mat_file:
            return os.path.abspath(mat_file)

    raise FileNotFoundError(
        f"Could not find any .mat data files for SNR={snr} in any searched location.\n"
        f"Searched roots: {candidate_roots}"
    )


def load_dataset_cgan(mat_filepath: str, input_type: str = 'LS') -> dict:
    """Load MATLAB v7/v7.3 channel grids formatted to [N, 132, 14, 2]."""
    input_type = input_type.upper()
    mat_dict = {}

    def format_3d(arr):
        if arr is None or not isinstance(arr, np.ndarray) or arr.ndim != 3:
            return arr
        if arr.shape[0] in (14, 132) and arr.shape[-1] not in (14, 132):
            arr = arr.T
        if arr.shape[1] == 14 and arr.shape[2] == 132:
            arr = np.transpose(arr, (0, 2, 1))
        return arr

    if h5py.is_hdf5(mat_filepath):
        with h5py.File(mat_filepath, 'r') as f:
            for k in f.keys():
                if not k.startswith('#'):
                    data = f[k][()]
                    if isinstance(data, np.ndarray) and data.dtype.names is not None:
                        if 'real' in data.dtype.names and 'imag' in data.dtype.names:
                            data = data['real'] + 1j * data['imag']
                    mat_dict[k] = data
    else:
        mat = loadmat(mat_filepath)
        for k, v in mat.items():
            if not k.startswith('__'):
                mat_dict[k] = v

    H_perfect = format_3d(mat_dict.get('H_perfect'))
    H_perfect_ori = None
    for k in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
        if k in mat_dict and mat_dict[k] is not None:
            H_perfect_ori = format_3d(mat_dict[k])
            break
    if H_perfect_ori is None:
        H_perfect_ori = H_perfect

    p_cols = np.squeeze(mat_dict['pilot_cols']).astype(int) - 1
    p_rows = np.squeeze(mat_dict['pilot_rows']).astype(int) - 1

    input_key_map = {'LS': 'H_LS', 'LI': 'H_LI', 'PRAC': 'H_prac'}
    target_key = input_key_map.get(input_type, 'H_LS')
    if target_key not in mat_dict or mat_dict[target_key] is None:
        for alt in ['H_LS', 'H_li', 'H_prac', 'H_perfect']:
            if alt in mat_dict and mat_dict[alt] is not None:
                target_key = alt
                break

    H_in = format_3d(mat_dict[target_key])
    H_li = format_3d(mat_dict.get('H_li', None))

    # Convert complex arrays to [N, 132, 14, 2] float32
    H_perf_real = np.stack([H_perfect.real, H_perfect.imag], axis=-1).astype(np.float32)
    H_in_real = np.stack([H_in.real, H_in.imag], axis=-1).astype(np.float32)

    return {
        'H_perfect': H_perfect,
        'H_perfect_ori': H_perfect_ori,
        'H_input': H_in,
        'H_li': H_li,
        'H_perfect_real': H_perf_real,
        'H_input_real': H_in_real,
        'pilot_cols': p_cols,
        'pilot_rows': p_rows,
        'N_samples': H_perfect.shape[0]
    }


def split_indices(N: int, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 1234):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def compute_nmse_db(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    diff_sq = np.sum(np.abs(H_pred - H_true)**2)
    ref_sq = np.sum(np.abs(H_true)**2)
    val = diff_sq / max(ref_sq, 1e-30)
    return float(10.0 * np.log10(val + 1e-30))


def compute_mmse(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    return float(np.mean(np.abs(H_pred - H_true)**2))


def compute_ssim_batch(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    real_pred = np.stack([H_pred.real, H_pred.imag], axis=-1).astype(np.float32)
    real_true = np.stack([H_true.real, H_true.imag], axis=-1).astype(np.float32)
    ssim_list = []
    for i in range(H_pred.shape[0]):
        dr = max(np.max(real_true[i]) - np.min(real_true[i]), 1e-12)
        s = tf_ssim(
            tf.convert_to_tensor(real_true[i:i+1]),
            tf.convert_to_tensor(real_pred[i:i+1]),
            max_val=dr
        )
        ssim_list.append(float(s.numpy()[0]))
    return float(np.mean(ssim_list))


def infer_full_dataset(generator, H_perf_real: np.ndarray, H_in_real: np.ndarray, batch_size: int = 32, lower_range: float = -1.0, standardize: bool = False):
    """Full-dataset batch inference on GPU with inverse MinMax or Standardization scaling."""
    N = H_in_real.shape[0]
    preds = []

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_raw = tf.convert_to_tensor(H_in_real[start:end], dtype=tf.float32)
        y_raw = tf.convert_to_tensor(H_perf_real[start:end], dtype=tf.float32)

        if standardize:
            x_sc, _, x_mean, x_std = batch_standardize(x_raw, y_raw)
            out_sc, _ = generator(x_sc, training=False)
            out_denorm = (out_sc * x_std + x_mean).numpy()
        else:
            x_sc, _, x_min, scale = batch_minmax_scale(x_raw, y_raw, lower_range)
            out_sc, _ = generator(x_sc, training=False)

            if lower_range == -1.0:
                out_norm = (out_sc + 1.0) / 2.0
            else:
                out_norm = out_sc

            out_denorm = (out_norm * scale + x_min).numpy()

        out_complex = out_denorm[..., 0] + 1j * out_denorm[..., 1]
        preds.append(out_complex)

    return np.concatenate(preds, axis=0)


def extract_features_cgan(generator, H_perf_real: np.ndarray, H_in_real: np.ndarray,
                          extract_layers: list, batch_size: int = 32, lower_range: float = -1.0, standardize: bool = False):
    """Extract intermediate features from UNet generator across full dataset."""
    N = H_in_real.shape[0]
    feats_dict = {lyr: [] for lyr in extract_layers}

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x_raw = tf.convert_to_tensor(H_in_real[start:end], dtype=tf.float32)
        y_raw = tf.convert_to_tensor(H_perf_real[start:end], dtype=tf.float32)

        if standardize:
            x_sc, _, _, _ = batch_standardize(x_raw, y_raw)
        else:
            x_sc, _, _, _ = batch_minmax_scale(x_raw, y_raw, lower_range)
        _, f_list = generator(x_sc, training=False, return_features=True)

        for lyr_name, f_t in zip(extract_layers, f_list):
            feats_dict[lyr_name].append(f_t.numpy())

    return {k: np.concatenate(v, axis=0) if len(v) > 0 else np.empty((0,)) for k, v in feats_dict.items()}


def save_test_channel_mat(filepath: str, H_perf: np.ndarray, H_perf_ori: np.ndarray,
                          H_in: np.ndarray, H_pred: np.ndarray, p_rows: np.ndarray,
                          p_cols: np.ndarray, H_li: np.ndarray, indices: np.ndarray,
                          snr: int, model_type: str):
    """Save test channel representations to MATLAB MAT file matching the standard schema."""
    out = {
        'H_perfect_test': H_perf,
        'H_original_test': H_perf_ori if H_perf_ori is not None else H_perf,
        'H_LS_test': H_in,
        'H_output_test': H_pred,
        'pilot_rows': p_rows + 1,
        'pilot_cols': p_cols + 1,
        'test_indices': indices,
        'snr': snr,
        'model_type': model_type
    }
    if H_li is not None:
        out['H_LI_test'] = H_li
    savemat(filepath, out)
    print(f"[Save] Exported test MAT file -> {filepath}")


# =============================================================================
# 5. VISUALIZATION HELPERS
# =============================================================================
def plot_loss_curves(history: dict, save_dir: str):
    epochs = range(1, len(history['train_g_loss']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_g_loss'], label='Generator Total Loss', color='blue', lw=2)
    if 'train_d_loss' in history:
        ax.plot(epochs, history['train_d_loss'], label='Discriminator Loss', color='orange', lw=1.5)
    if 'train_jmmd_loss' in history:
        ax.plot(epochs, history['train_jmmd_loss'], label='JMMD Domain Loss', color='red', lw=1.5, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('JMMD cGAN Training Loss Progression')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, 'loss_total.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_val_curves(history: dict, save_dir: str):
    if not history.get('val_nmse_tgt'):
        return
    epochs = range(1, len(history['val_nmse_tgt']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'val_nmse_src' in history:
        ax.plot(epochs, history['val_nmse_src'], label='Source Val NMSE', color='blue', lw=1.5)
    ax.plot(epochs, history['val_nmse_tgt'], label='Target Val NMSE', color='orange', lw=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('Validation NMSE (dB) Across Epochs')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, 'val_nmse_db.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)


def save_channel_plots_pdf(H_true_sample, H_in_sample, H_pred_sample, save_dir, prefix='target_test'):
    """Side-by-side visualization PDF."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].imshow(np.abs(H_true_sample), aspect='auto', cmap='jet')
    axes[0].set_title("Ground Truth Channel |H|")
    axes[0].set_xlabel("OFDM Symbol")
    axes[0].set_ylabel("Subcarrier")
    plt.colorbar(im0, ax=axes[0])

    if H_in_sample.ndim == 2 and H_in_sample.shape == (132, 14):
        im1 = axes[1].imshow(np.abs(H_in_sample), aspect='auto', cmap='jet')
    else:
        im1 = axes[1].plot(np.abs(H_in_sample))
    axes[1].set_title("Input LS Pilot |H_in|")

    im2 = axes[2].imshow(np.abs(H_pred_sample), aspect='auto', cmap='jet')
    axes[2].set_title("cGAN Estimated Channel |H_pred|")
    axes[2].set_xlabel("OFDM Symbol")
    axes[2].set_ylabel("Subcarrier")
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    out_pdf = os.path.join(save_dir, f'{prefix}_sample_reconstruction.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Save] Exported reconstruction plot -> {out_pdf}")


# =============================================================================
# 6. MAIN EXECUTION PIPELINE
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="High-Performance JMMD cGAN Domain Adaptation for OpenNTN Channel Estimation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source-dir', type=str, default=DEFAULT_SOURCE_DIR, help="Source dataset directory or MAT path")
    parser.add_argument('--target-dir', type=str, default=DEFAULT_TARGET_DIR, help="Target dataset directory or MAT path")
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR, help="Directory to save output files")
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, help="Channel SNR in dB")
    parser.add_argument('--type', type=str, default=DEFAULT_TYPE, choices=['LS', 'LI', 'Prac', 'ls', 'li', 'prac'], help="Input type")
    parser.add_argument('--only-source', action='store_true', default=DEFAULT_ONLY_SOURCE, help="Train only on source data (no JMMD)")
    parser.add_argument('--jmmd-layers', nargs='+', default=DEFAULT_JMMD_LAYERS, help="UNet layers to extract for JMMD (e.g. 'd4' or 'd3 d4')")
    parser.add_argument('--jmmd-mode', type=str, default=DEFAULT_JMMD_MODE, choices=['joint', 'layerwise'], help="JMMD kernel mode")
    parser.add_argument('--kernel-num', type=int, default=DEFAULT_KERNEL_NUM, help="Number of Gaussian RBF kernels")
    parser.add_argument('--kernel-mul', type=float, default=DEFAULT_KERNEL_MUL, help="Gaussian kernel multiplier")
    parser.add_argument('--adv-weight', type=float, default=DEFAULT_ADV_WEIGHT, help="WGAN adversarial loss weight")
    parser.add_argument('--est-weight', type=float, default=DEFAULT_EST_WEIGHT, help="Estimation L1 loss weight")
    parser.add_argument('--domain-weight', type=float, default=DEFAULT_DOMAIN_WEIGHT, help="JMMD domain alignment loss weight")
    parser.add_argument('--temporal-weight', type=float, default=DEFAULT_TEMPORAL_WEIGHT, help="Temporal smoothness weight")
    parser.add_argument('--frequency-weight', type=float, default=DEFAULT_FREQUENCY_WEIGHT, help="Frequency smoothness weight")
    parser.add_argument('--ssim-weight', type=float, default=DEFAULT_SSIM_WEIGHT, help="SSIM structural loss weight")
    parser.add_argument('--gp-weight', type=float, default=DEFAULT_GP_WEIGHT, help="WGAN-GP gradient penalty weight")
    parser.add_argument('--n-epochs', type=int, default=DEFAULT_N_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    parser.add_argument('--lr-g', type=float, default=DEFAULT_LR_G, help="Generator learning rate")
    parser.add_argument('--lr-d', type=float, default=DEFAULT_LR_D, help="Discriminator learning rate")
    parser.add_argument('--lower-range', type=int, default=-1, choices=[-1, 0], help="Min-max scaling lower range")
    parser.add_argument('--standardize', action='store_true', default=False, help="Use sample-wise zero-mean unit-variance standardization instead of min-max scaling")
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help="Fraction of data for training")
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help="Fraction of data for validation")
    parser.add_argument('--save-features', action='store_true', default=DEFAULT_SAVE_FEATURES, help="Extract & save intermediate features")
    parser.add_argument('--test-code', action='store_true', help="Fast sanity test run (5 epochs, small subset)")
    parser.add_argument('--no-gpu', action='store_true', help="Disable GPU execution")

    args = parser.parse_args()

    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("[Config] GPU disabled by user.")

    # Parse and clean JMMD layer list
    extract_layers = []
    for item in args.jmmd_layers:
        for sub in str(item).replace(',', ' ').split():
            clean_sub = sub.strip().lower()
            if clean_sub and clean_sub not in extract_layers:
                extract_layers.append(clean_sub)

    valid_layers = ['d1', 'd2', 'd3', 'd4', 'u1', 'u2', 'u3']
    for lyr in extract_layers:
        if lyr not in valid_layers:
            raise ValueError(f"Invalid layer '{lyr}' in --jmmd-layers. Valid options are: {valid_layers}")

    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError(f"train_frac ({args.train_frac}) + val_frac ({args.val_frac}) must be < 1.0.")
    test_frac = 1.0 - args.train_frac - args.val_frac

    args.type = args.type.upper()
    domain_weight = 0.0 if args.only_source else args.domain_weight

    # Resolve dataset paths dynamically
    src_mat_path = os.path.abspath(get_mat_file(args.source_dir, args.snr))
    tgt_mat_path = os.path.abspath(get_mat_file(args.target_dir, args.snr))

    print("=" * 80)
    print(f"JMMD cGAN Model | Mode: {'Source-Only' if args.only_source else f'JMMD ({args.jmmd_mode}) UDA'}")
    if not args.only_source:
        print(f"JMMD Extracted Layers: {extract_layers} ({len(extract_layers)} UNet layers)")
        print(f"Gaussian Kernels: {args.kernel_num} scales (mul={args.kernel_mul})")
        print(f"JMMD Loss Weight (lambda): {domain_weight}")
    norm_str = "Standardize (Zero-Mean, Unit-Var)" if args.standardize else f"Min-Max [{args.lower_range}, 1]"
    print(f"Source Dataset: {src_mat_path}")
    print(f"Target Dataset: {tgt_mat_path}")
    print(f"SNR: {args.snr} dB | Input Type: {args.type} | Normalization: {norm_str}")
    print(f"Split: {args.train_frac:.0%} Train / {args.val_frac:.0%} Val / {test_frac:.0%} Test")
    print("=" * 80)

    # Save Directory Setup
    if args.save_dir and args.save_dir.strip():
        output_dir = os.path.abspath(args.save_dir)
    else:
        output_dir = os.path.join(current_dir, 'results')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment results will be saved to: {output_dir}")

    # Load Source and Target Datasets
    src_data = load_dataset_cgan(src_mat_path, args.type)
    tgt_data = load_dataset_cgan(tgt_mat_path, args.type)

    N_src = src_data['N_samples']
    N_tgt = tgt_data['N_samples']

    # 3-Way Split Indices
    if args.test_code:
        idx_train_src, idx_val_src, idx_test_src = np.arange(0, 64), np.arange(64, 80), np.arange(80, 96)
        idx_train_tgt, idx_val_tgt, idx_test_tgt = np.arange(0, 64), np.arange(64, 80), np.arange(80, 96)
        args.n_epochs = 5
    else:
        idx_train_src, idx_val_src, idx_test_src = split_indices(N_src, args.train_frac, args.val_frac, seed=1234)
        idx_train_tgt, idx_val_tgt, idx_test_tgt = split_indices(N_tgt, args.train_frac, args.val_frac, seed=1234)

    print(f"Source Split -> Train: {len(idx_train_src)} | Val: {len(idx_val_src)} | Test: {len(idx_test_src)}")
    print(f"Target Split -> Train: {len(idx_train_tgt)} | Val: {len(idx_val_tgt)} | Test: {len(idx_test_tgt)}")

    # Instantiate Models & Optimizers
    generator = Pix2PixGenerator(output_channels=2, extract_layers=extract_layers)
    discriminator = PatchGANDiscriminator(filters=[32, 64, 128, 256])

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_g, beta_1=0.5, beta_2=0.9)
    disc_optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr_d, beta_1=0.5, beta_2=0.9)

    # Pre-build model variables with a dummy forward pass
    dummy_input = tf.zeros([args.batch_size, 132, 14, 2], dtype=tf.float32)
    dummy_out, dummy_feats = generator(dummy_input, training=False, return_features=True)
    _ = discriminator(dummy_out, training=False)

    print(f"\n[Model Initialized] UNet Generator + PatchGAN Discriminator (WGAN-GP)")
    print(f"  Extracting JMMD Layers: {extract_layers}")

    history = {
        'train_g_loss': [],
        'train_d_loss': [],
        'train_est_loss': [],
        'train_jmmd_loss': [],
        'val_nmse_src': [],
        'val_nmse_tgt': []
    }

    saved_features = {}
    mid_epoch = args.n_epochs // 2
    feature_checkpoint_epochs = {0: 'begin', mid_epoch: 'mid', args.n_epochs - 1: 'last'}

    # Prepare GPU-ready Tensors
    src_train_in = src_data['H_input_real'][idx_train_src]
    src_train_perf = src_data['H_perfect_real'][idx_train_src]
    tgt_train_in = tgt_data['H_input_real'][idx_train_tgt]
    tgt_train_perf = tgt_data['H_perfect_real'][idx_train_tgt]

    src_val_in = src_data['H_input_real'][idx_val_src]
    src_val_perf = src_data['H_perfect_real'][idx_val_src]
    tgt_val_in = tgt_data['H_input_real'][idx_val_tgt]
    tgt_val_perf = tgt_data['H_perfect_real'][idx_val_tgt]

    n_train_samples = min(len(idx_train_src), len(idx_train_tgt))
    n_batches = n_train_samples // args.batch_size

    print(f"\n[Train] Starting JMMD cGAN Training for {args.n_epochs} Epochs ({n_batches} batches/epoch) ...")
    start_time = time.perf_counter()

    ssim_max_val = 6.0 if args.standardize else (2.0 if args.lower_range == -1 else 1.0)

    for epoch in range(args.n_epochs):
        # Feature Checkpointing at begin, mid, and last epoch
        if args.save_features and epoch in feature_checkpoint_epochs:
            stage = feature_checkpoint_epochs[epoch]
            src_feats = extract_features_cgan(generator, src_train_perf, src_train_in, extract_layers, args.batch_size, float(args.lower_range), standardize=args.standardize)
            tgt_feats = extract_features_cgan(generator, tgt_train_perf, tgt_train_in, extract_layers, args.batch_size, float(args.lower_range), standardize=args.standardize)
            for lyr in extract_layers:
                saved_features[f"features_{stage}_{lyr}_src"] = src_feats[lyr]
                saved_features[f"features_{stage}_{lyr}_tgt"] = tgt_feats[lyr]
            print(f"  [Features Saved] Captured intermediate activations at {stage} epoch ({epoch+1}) for layers: {extract_layers}")

        # Shuffle training sets
        p_src = np.random.permutation(len(idx_train_src))
        p_tgt = np.random.permutation(len(idx_train_tgt))

        epoch_g_loss, epoch_d_loss, epoch_est_loss, epoch_jmmd_loss = 0.0, 0.0, 0.0, 0.0

        for b in range(n_batches):
            s_idx = b * args.batch_size
            e_idx = s_idx + args.batch_size

            # In-memory batch slices
            bx_src = tf.convert_to_tensor(src_train_in[p_src[s_idx:e_idx]], dtype=tf.float32)
            by_src = tf.convert_to_tensor(src_train_perf[p_src[s_idx:e_idx]], dtype=tf.float32)
            bx_tgt = tf.convert_to_tensor(tgt_train_in[p_tgt[s_idx:e_idx]], dtype=tf.float32)
            by_tgt = tf.convert_to_tensor(tgt_train_perf[p_tgt[s_idx:e_idx]], dtype=tf.float32)

            # GPU Scaling / Standardization
            if args.standardize:
                x_src_sc, y_src_sc, _, _ = batch_standardize(bx_src, by_src)
                x_tgt_sc, _, _, _ = batch_standardize(bx_tgt, by_tgt)
            else:
                x_src_sc, y_src_sc, _, _ = batch_minmax_scale(bx_src, by_src, float(args.lower_range))
                x_tgt_sc, _, _, _ = batch_minmax_scale(bx_tgt, by_tgt, float(args.lower_range))

            if args.only_source:
                g_loss, d_loss, est_loss, jmmd_loss, _ = train_step_source_only(
                    generator, discriminator, gen_optimizer, disc_optimizer,
                    x_src_sc, y_src_sc,
                    args.adv_weight, args.est_weight, args.temporal_weight,
                    args.frequency_weight, args.ssim_weight, args.gp_weight,
                    ssim_max_val=ssim_max_val
                )
            else:
                g_loss, d_loss, est_loss, jmmd_loss, _ = train_step_jmmd(
                    generator, discriminator, gen_optimizer, disc_optimizer,
                    x_src_sc, y_src_sc, x_tgt_sc,
                    args.adv_weight, args.est_weight, domain_weight, args.temporal_weight,
                    args.frequency_weight, args.ssim_weight, args.gp_weight,
                    kernel_mul=args.kernel_mul, kernel_num=args.kernel_num, jmmd_mode=args.jmmd_mode,
                    ssim_max_val=ssim_max_val
                )

            epoch_g_loss += float(g_loss)
            epoch_d_loss += float(d_loss)
            epoch_est_loss += float(est_loss)
            epoch_jmmd_loss += float(jmmd_loss)

        avg_g_loss = epoch_g_loss / max(n_batches, 1)
        avg_d_loss = epoch_d_loss / max(n_batches, 1)
        avg_est_loss = epoch_est_loss / max(n_batches, 1)
        avg_jmmd_loss = epoch_jmmd_loss / max(n_batches, 1)

        history['train_g_loss'].append(avg_g_loss)
        history['train_d_loss'].append(avg_d_loss)
        history['train_est_loss'].append(avg_est_loss)
        history['train_jmmd_loss'].append(avg_jmmd_loss)

        # Periodic Validation
        pred_val_src = infer_full_dataset(generator, src_val_perf, src_val_in, args.batch_size, float(args.lower_range), standardize=args.standardize)
        val_nmse_src_db = compute_nmse_db(pred_val_src, src_data['H_perfect'][idx_val_src])
        history['val_nmse_src'].append(val_nmse_src_db)

        pred_val_tgt = infer_full_dataset(generator, tgt_val_perf, tgt_val_in, args.batch_size, float(args.lower_range), standardize=args.standardize)
        val_nmse_tgt_db = compute_nmse_db(pred_val_tgt, tgt_data['H_perfect'][idx_val_tgt])
        history['val_nmse_tgt'].append(val_nmse_tgt_db)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.n_epochs - 1:
            print(f"Epoch {epoch+1:03d}/{args.n_epochs:03d} | G-Loss: {avg_g_loss:.4f} (Est: {avg_est_loss:.4f}, JMMD: {avg_jmmd_loss:.4f}) | D-Loss: {avg_d_loss:.4f} | Val Target NMSE: {val_nmse_tgt_db:.2f} dB")

    total_time = time.perf_counter() - start_time
    print(f"\n[Done] Training completed in {total_time:.2f} seconds ({total_time / args.n_epochs:.3f} s/epoch).")

    # =========================================================================
    # FINAL TEST EVALUATION & EXPORTS
    # =========================================================================
    print("\n" + "=" * 80)
    print("                      FINAL TEST PERFORMANCE SUMMARY                  ")
    print("=" * 80)

    # 1. Source Test Evaluation
    src_test_perf = src_data['H_perfect_real'][idx_test_src]
    src_test_in = src_data['H_input_real'][idx_test_src]
    test_pred_src = infer_full_dataset(generator, src_test_perf, src_test_in, args.batch_size, float(args.lower_range), standardize=args.standardize)
    test_nmse_db_src = compute_nmse_db(test_pred_src, src_data['H_perfect'][idx_test_src])
    test_mmse_src = compute_mmse(test_pred_src, src_data['H_perfect'][idx_test_src])
    test_ssim_src = compute_ssim_batch(test_pred_src, src_data['H_perfect'][idx_test_src])
    print(f"  Source Domain -> NMSE: {test_nmse_db_src:.2f} dB | MMSE: {test_mmse_src:.6e} | SSIM: {test_ssim_src:.4f}")

    # 2. Target Test Evaluation
    tgt_test_perf = tgt_data['H_perfect_real'][idx_test_tgt]
    tgt_test_in = tgt_data['H_input_real'][idx_test_tgt]
    test_pred_tgt = infer_full_dataset(generator, tgt_test_perf, tgt_test_in, args.batch_size, float(args.lower_range), standardize=args.standardize)
    test_nmse_db_tgt = compute_nmse_db(test_pred_tgt, tgt_data['H_perfect'][idx_test_tgt])
    test_mmse_tgt = compute_mmse(test_pred_tgt, tgt_data['H_perfect'][idx_test_tgt])
    test_ssim_tgt = compute_ssim_batch(test_pred_tgt, tgt_data['H_perfect'][idx_test_tgt])
    print(f"  Target Domain -> NMSE: {test_nmse_db_tgt:.2f} dB | MMSE: {test_mmse_tgt:.6e} | SSIM: {test_ssim_tgt:.4f}")
    print("=" * 80)

    # 3. Save testChannel_source.mat and testChannel_target.mat
    save_test_channel_mat(
        os.path.join(output_dir, 'testChannel_source.mat'),
        src_data['H_perfect'][idx_test_src],
        src_data['H_perfect_ori'][idx_test_src] if src_data['H_perfect_ori'] is not None else None,
        src_data['H_input'][idx_test_src],
        test_pred_src,
        src_data['pilot_rows'], src_data['pilot_cols'],
        src_data['H_li'][idx_test_src] if src_data['H_li'] is not None else None,
        idx_test_src, args.snr, args.type
    )

    save_test_channel_mat(
        os.path.join(output_dir, 'testChannel_target.mat'),
        tgt_data['H_perfect'][idx_test_tgt],
        tgt_data['H_perfect_ori'][idx_test_tgt] if tgt_data['H_perfect_ori'] is not None else None,
        tgt_data['H_input'][idx_test_tgt],
        test_pred_tgt,
        tgt_data['pilot_rows'], tgt_data['pilot_cols'],
        tgt_data['H_li'][idx_test_tgt] if tgt_data['H_li'] is not None else None,
        idx_test_tgt, args.snr, args.type
    )

    # 4. Save final_epoch.txt Report
    txt_path = os.path.join(output_dir, 'final_epoch.txt')
    try:
        with open(txt_path, 'w') as f:
            f.write("=== FINAL EPOCH EVALUATION RESULTS ===\n")
            f.write(f"SNR (dB):             {args.snr}\n")
            f.write(f"Input Type:           {args.type}\n")
            f.write(f"Domain Adaptation:    {'Source-Only' if args.only_source else f'JMMD cGAN ({args.jmmd_mode}) UDA'}\n")
            f.write(f"Normalization:        {norm_str}\n")
            f.write(f"JMMD Layers:          {extract_layers}\n")
            f.write(f"Gaussian Kernels:     {args.kernel_num} scales (mul={args.kernel_mul})\n")
            f.write(f"Total Execution Time: {total_time:.1f} s\n\n")

            f.write("--- SOURCE DOMAIN TEST METRICS ---\n")
            f.write(f"Model Output MMSE:    {test_mmse_src:e}\n")
            f.write(f"Model Output NMSE:    {compute_nmse_db(test_pred_src, src_data['H_perfect'][idx_test_src]):.2f} dB\n")
            f.write(f"Model Output SSIM:    {test_ssim_src:.4f}\n\n")

            f.write("--- TARGET DOMAIN TEST METRICS ---\n")
            f.write(f"Model Output MMSE:    {test_mmse_tgt:e}\n")
            f.write(f"Model Output NMSE:    {compute_nmse_db(test_pred_tgt, tgt_data['H_perfect'][idx_test_tgt]):.2f} dB\n")
            f.write(f"Model Output SSIM:    {test_ssim_tgt:.4f}\n")
        print(f"[Save] Final epoch text report -> {txt_path}")
    except Exception as e:
        print(f"[Save Warning] Failed to write final_epoch.txt: {e}")

    # 5. Save evaluation_results.mat
    eval_path = os.path.join(output_dir, 'evaluation_results.mat')
    eval_dict = {
        'nmse_test_src_db': test_nmse_db_src,
        'mmse_test_src': test_mmse_src,
        'ssim_test_src': test_ssim_src,
        'nmse_test_tgt_db': test_nmse_db_tgt,
        'mmse_test_tgt': test_mmse_tgt,
        'ssim_test_tgt': test_ssim_tgt,
        'mmse_test': test_mmse_tgt,
        'nmse_test_db': test_nmse_db_tgt,
        'ssim_test': test_ssim_tgt,
        'indices_train_source': idx_train_src,
        'indices_val_source': idx_val_src,
        'indices_test_source': idx_test_src,
        'indices_train_target': idx_train_tgt,
        'indices_val_target': idx_val_tgt,
        'indices_test_target': idx_test_tgt,
        'snr': args.snr,
        'input_type': args.type,
        'standardize': args.standardize,
        'jmmd_mode': args.jmmd_mode,
        'jmmd_layers': np.array(extract_layers)
    }
    savemat(eval_path, eval_dict)
    print(f"[Save] Evaluation results -> {eval_path}")

    # 6. Save Training History
    history_save_path = os.path.join(output_dir, 'training_history.mat')
    savemat(history_save_path, {k: np.array(v) for k, v in history.items()})
    print(f"[Save] Saved training history -> {history_save_path}")

    # 7. Save Extracted Features MAT if requested
    if args.save_features and saved_features:
        feat_save_path = os.path.join(output_dir, 'extracted_features.mat')
        saved_features['selected_layers'] = np.array(extract_layers)
        saved_features['train_indices_src'] = idx_train_src
        saved_features['train_indices_tgt'] = idx_train_tgt
        savemat(feat_save_path, saved_features, do_compression=True)
        print(f"[Save] Exported extracted features -> {feat_save_path}")

    # 8. Save Visualizations
    try:
        plot_loss_curves(history, output_dir)
        plot_val_curves(history, output_dir)
        if len(test_pred_tgt) > 0:
            sample_true = tgt_data['H_perfect'][idx_test_tgt[0]]
            sample_in = tgt_data['H_input'][idx_test_tgt[0]]
            save_channel_plots_pdf(sample_true, sample_in, test_pred_tgt[0], output_dir, prefix='target_test')
    except Exception as e:
        print(f"[Plot Warning] Failed to render PDF plots: {e}")

    print(f"\n[Done] Finished JMMD cGAN training and evaluation. Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
