"""
====================================================================================================
CORAL cGAN Domain Adaptation for NTN Channel Estimation (OpenNTN) - High-Performance Edition
====================================================================================================

Overview
--------
This script trains and evaluates a Conditional Generative Adversarial Network (cGAN) based on 
Pix2Pix (UNet Generator + PatchGAN Discriminator with WGAN-GP) to perform Unsupervised Domain 
Adaptation (UDA) for 5G Non-Terrestrial Network (NTN) channel estimation.

Domain shift (e.g., between different user speeds, propagation delays, or TDL channel profiles)
is mitigated using Correlation Alignment (CORAL) loss, which aligns the second-order statistics
(covariance matrices) of intermediate feature representations between the source and target domains.

Performance Optimizations
-------------------------
- Pure `@tf.function` Graph Execution: Compiled CUDA kernels execute WGAN-GP, CORAL, and Generator
  steps with zero Python eager-mode dispatch overhead.
- In-Memory GPU Batching: Zero per-batch CPU array copies or repeated transposition.
- Fast Vectorized Validation & SSIM: Skips redundant discriminator passes during validation.

Dataset Splitting (3-Way Split)
--------------------------------
- Train Set (Default 70%): Labeled source domain + Unlabeled target domain for CORAL adaptation.
- Validation Set (Default 15%): Periodic evaluation and model checkpointing during training.
- Test Set (Default 15%): Final held-out evaluation exported to `testChannel_source.mat` and 
  `testChannel_target.mat` for subsequent BER simulations and benchmark comparisons.

Usage Examples
--------------
    # Default direct estimation cGAN with CORAL adaptation (70/15/15 split at 5 dB)
    python run_CORAL_cGAN.py --type LI --snr 5 --domain-weight 0.5

    # Multi-layer CORAL feature alignment (e.g. d3 and d4 bottleneck)
    python run_CORAL_cGAN.py --type LI --snr 5 --coral-layers d3 d4 --save-features

    # Source-only baseline (no domain adaptation)
    python run_CORAL_cGAN.py --type LI --snr 5 --only-source

    # Residual learning mode
    python run_CORAL_cGAN.py --type LI --snr 0 --residual

    # Quick test run (small subset, 5 epochs)
    python run_CORAL_cGAN.py --type LI --test-code
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

import tensorflow as tf
from tensorflow.image import ssim as tf_ssim
import os
import sys
import scipy
from scipy.io import savemat, loadmat
import h5py
import time
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================
SOURCE_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_20to30mps"
TARGET_DIR = r"C:\Users\AT30890\Hoctap\1_Hprediction\working\H_predict_NTN\Hest_NTN_UDA_clean\generatedChan\OpenNTN\DUR100nsFix_2p18G_600km_70deg_r15km_30to40mps"
DEFAULT_SAVE_DIR = ""          # Output save directory (defaults to './results' inside current directory)
DEFAULT_SNR = 5                # Channel SNR in dB
MODEL_TYPE = "LI"              # "LI", "LS", or "Prac"
ONLY_SOURCE = False             # Set True to train only on source (no CORAL)
RESIDUAL = False                # Set True for residual cGAN, False for direct estimation cGAN
CORAL_LAYERS = ["d2", "d3", "d4"]  # Extracted layers for CORAL (e.g. ['d3', 'd4'] or ['d4'])
SAVE_FEATURES = False           # Set True to save extracted features at begin, mid, last epochs
DEFAULT_TRAIN_FRAC = 0.70       # Fraction of data for training
DEFAULT_VAL_FRAC = 0.15         # Fraction of data for validation (remaining is test)
N_EPOCHS = 300                  # Number of epochs
BATCH_SIZE = 16                 # Batch size
TEST_CODE = False               # Run with subset of data for quick testing
# ============================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))


# ============================================================================
# NEURAL NETWORK ARCHITECTURE DEFINITIONS (cGAN Pix2Pix)
# ============================================================================

class InstanceNormalization(tf.keras.layers.Layer):
    """Instance Normalization Layer for Style/Texture Normalization."""
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
            name="gamma"
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
            name="beta"
        )

    def call(self, x, training=False):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_norm = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * x_norm + self.beta


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
    """UNet Generator for 5G Channel Estimation."""
    def __init__(self, output_channels=2, gen_l2=None,
                 dropOut_layers=['u1', 'u2'], dropOut_rate=0.3, extract_layers=['d2', 'd3', 'd4']):
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


# ============================================================================
# FAST VECTORIZED LOSS FUNCTIONS & GPU NORMALIZATION
# ============================================================================

@tf.function
def batch_minmax_scale(x, lower_range=-1.0):
    """Vectorized GPU Sample-wise MinMax Normalization along subcarrier and symbol axes."""
    x_min = tf.reduce_min(x, axis=[1, 2], keepdims=True)
    x_max = tf.reduce_max(x, axis=[1, 2], keepdims=True)
    scale = tf.clip_by_value(x_max - x_min, 1e-12, tf.float32.max)
    x_norm = (x - x_min) / scale
    if lower_range == -1.0:
        x_norm = x_norm * 2.0 - 1.0
    return x_norm, x_min, x_max


@tf.function
def batch_minmax_scale_fixed(y, x_min, x_max, lower_range=-1.0):
    """Normalize target channel using predefined min/max scaling parameters."""
    scale = tf.clip_by_value(x_max - x_min, 1e-12, tf.float32.max)
    y_norm = (y - x_min) / scale
    if lower_range == -1.0:
        y_norm = y_norm * 2.0 - 1.0
    return y_norm


@tf.function
def batch_minmax_descale(x_norm, x_min, x_max, lower_range=-1.0):
    """Vectorized GPU Descaling back to original physical channel scale."""
    if lower_range == -1.0:
        x_norm = (x_norm + 1.0) * 0.5
    scale = x_max - x_min
    return x_norm * scale + x_min


@tf.function
def compute_gradient_penalty(discriminator, real, fake):
    """WGAN-GP Gradient Penalty computed entirely on GPU."""
    batch_size = tf.shape(real)[0]
    alpha = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0, dtype=tf.float32)
    interpolated = alpha * real + (1.0 - alpha) * fake
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolated)
        pred = discriminator(interpolated, training=True)
    grads = gp_tape.gradient(pred, interpolated)
    norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]) + 1e-12)
    return tf.reduce_mean(tf.square(norm - 1.0))


@tf.function
def compute_coral_loss(features_src, features_tgt):
    """Vectorized Global Average Pooling CORAL Loss across extracted layer representations."""
    total_loss = 0.0
    n_layers = tf.cast(len(features_src), tf.float32)
    if n_layers == 0.0:
        return tf.constant(0.0, dtype=tf.float32)

    for f_s, f_t in zip(features_src, features_tgt):
        if len(f_s.shape) == 4:
            f_s = tf.reduce_mean(f_s, axis=[1, 2])
            f_t = tf.reduce_mean(f_t, axis=[1, 2])
        elif len(f_s.shape) > 2:
            f_s = tf.reshape(f_s, [tf.shape(f_s)[0], -1])
            f_t = tf.reshape(f_t, [tf.shape(f_t)[0], -1])

        n_s = tf.cast(tf.shape(f_s)[0], tf.float32)
        n_t = tf.cast(tf.shape(f_t)[0], tf.float32)
        d = tf.cast(tf.shape(f_s)[1], tf.float32)

        f_s_c = f_s - tf.reduce_mean(f_s, axis=0, keepdims=True)
        f_t_c = f_t - tf.reduce_mean(f_t, axis=0, keepdims=True)

        cov_s = tf.matmul(f_s_c, f_s_c, transpose_a=True) / tf.maximum(n_s - 1.0, 1.0)
        cov_t = tf.matmul(f_t_c, f_t_c, transpose_a=True) / tf.maximum(n_t - 1.0, 1.0)

        layer_coral = tf.reduce_sum(tf.square(cov_s - cov_t)) / (4.0 * d * d)
        total_loss += layer_coral

    return total_loss / n_layers


@tf.function
def compute_smoothness_loss(x, temporal_weight=0.02, frequency_weight=0.1):
    """Temporal and Frequency Smoothness Regularization."""
    if temporal_weight == 0.0 and frequency_weight == 0.0:
        return tf.constant(0.0, dtype=tf.float32)
    t_loss = tf.reduce_mean(tf.square(x[:, :, 1:, :] - x[:, :, :-1, :]))
    f_loss = tf.reduce_mean(tf.square(x[:, 1:, :, :] - x[:, :-1, :, :]))
    return temporal_weight * t_loss + frequency_weight * f_loss


# ============================================================================
# COMPILED HIGH-PERFORMANCE TRAINING & VALIDATION STEPS
# ============================================================================

@tf.function
def train_step_coral(generator, discriminator, opt_gen, opt_disc,
                     x_src_raw, y_src_raw, x_tgt_raw, y_tgt_raw,
                     adv_weight=0.005, est_weight=1.0, domain_weight=0.5,
                     temporal_weight=0.02, frequency_weight=0.1, is_residual=False, lower_range=-1.0):
    """Fully fused and compiled cGAN + CORAL training step."""
    # Fast GPU Normalization
    x_src, x_min_s, x_max_s = batch_minmax_scale(x_src_raw, lower_range)
    y_src = batch_minmax_scale_fixed(y_src_raw, x_min_s, x_max_s, lower_range)
    x_tgt, x_min_t, x_max_t = batch_minmax_scale(x_tgt_raw, lower_range)
    y_tgt = batch_minmax_scale_fixed(y_tgt_raw, x_min_t, x_max_t, lower_range)

    # 1. Train Discriminator (WGAN-GP on source domain)
    with tf.GradientTape() as tape_d:
        fake_src_norm, _ = generator(x_src, training=True)
        if is_residual:
            fake_src_norm = x_src + fake_src_norm

        d_real = discriminator(y_src, training=True)
        d_fake = discriminator(fake_src_norm, training=True)
        gp = compute_gradient_penalty(discriminator, y_src, fake_src_norm)
        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + 10.0 * gp
        if discriminator.losses:
            d_loss += tf.add_n(discriminator.losses)

    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    opt_disc.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    # 2. Train Generator (Estimation + Adversarial + CORAL + Smoothness)
    with tf.GradientTape() as tape_g:
        fake_src_norm, feats_src = generator(x_src, training=True)
        fake_tgt_norm, feats_tgt = generator(x_tgt, training=True)

        out_src = x_src + fake_src_norm if is_residual else fake_src_norm
        out_tgt = x_tgt + fake_tgt_norm if is_residual else fake_tgt_norm

        est_loss = tf.reduce_mean(tf.square(y_src - out_src))
        est_loss_tgt = tf.reduce_mean(tf.square(y_tgt - out_tgt))

        d_fake_g = discriminator(out_src, training=True)
        adv_loss = -tf.reduce_mean(d_fake_g)

        coral_loss = compute_coral_loss(feats_src, feats_tgt) if domain_weight > 0 else tf.constant(0.0, dtype=tf.float32)
        smooth_loss = (compute_smoothness_loss(out_src, temporal_weight, frequency_weight) +
                       compute_smoothness_loss(out_tgt, temporal_weight, frequency_weight)) * 0.5

        g_loss = est_weight * est_loss + adv_weight * adv_loss + domain_weight * coral_loss + smooth_loss
        if generator.losses:
            g_loss += tf.add_n(generator.losses)

    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    opt_gen.apply_gradients(zip(grads_g, generator.trainable_variables))

    return g_loss, est_loss, d_loss, coral_loss, est_loss_tgt


@tf.function
def train_step_source_only(generator, discriminator, opt_gen, opt_disc,
                           x_src_raw, y_src_raw,
                           adv_weight=0.005, est_weight=1.0,
                           temporal_weight=0.02, frequency_weight=0.1, is_residual=False, lower_range=-1.0):
    """Fully fused and compiled Source-Only cGAN training step."""
    x_src, x_min_s, x_max_s = batch_minmax_scale(x_src_raw, lower_range)
    y_src = batch_minmax_scale_fixed(y_src_raw, x_min_s, x_max_s, lower_range)

    # 1. Train Discriminator
    with tf.GradientTape() as tape_d:
        fake_src_norm, _ = generator(x_src, training=True)
        if is_residual:
            fake_src_norm = x_src + fake_src_norm

        d_real = discriminator(y_src, training=True)
        d_fake = discriminator(fake_src_norm, training=True)
        gp = compute_gradient_penalty(discriminator, y_src, fake_src_norm)
        d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real) + 10.0 * gp
        if discriminator.losses:
            d_loss += tf.add_n(discriminator.losses)

    grads_d = tape_d.gradient(d_loss, discriminator.trainable_variables)
    opt_disc.apply_gradients(zip(grads_d, discriminator.trainable_variables))

    # 2. Train Generator
    with tf.GradientTape() as tape_g:
        fake_src_norm, _ = generator(x_src, training=True)
        out_src = x_src + fake_src_norm if is_residual else fake_src_norm

        est_loss = tf.reduce_mean(tf.square(y_src - out_src))
        d_fake_g = discriminator(out_src, training=True)
        adv_loss = -tf.reduce_mean(d_fake_g)
        smooth_loss = compute_smoothness_loss(out_src, temporal_weight, frequency_weight)

        g_loss = est_weight * est_loss + adv_weight * adv_loss + smooth_loss
        if generator.losses:
            g_loss += tf.add_n(generator.losses)

    grads_g = tape_g.gradient(g_loss, generator.trainable_variables)
    opt_gen.apply_gradients(zip(grads_g, generator.trainable_variables))

    return g_loss, est_loss, d_loss, tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32)


@tf.function
def val_step_fast(generator, x_raw, y_raw, is_residual=False, lower_range=-1.0):
    """Fast compiled validation step measuring exact squared error and reference power on GPU."""
    x_norm, x_min, x_max = batch_minmax_scale(x_raw, lower_range)
    pred_norm, _ = generator(x_norm, training=False)
    pred_descaled = batch_minmax_descale(pred_norm, x_min, x_max, lower_range)
    if is_residual:
        pred_descaled = x_raw + pred_descaled

    err_sq = tf.reduce_sum(tf.square(y_raw - pred_descaled))
    ref_sq = tf.reduce_sum(tf.square(y_raw))
    return err_sq, ref_sq


# ============================================================================
# DATASET LOADING & METRIC HELPERS
# ============================================================================

def get_mat_file(dir_path: str, snr: int = 5) -> str:
    """Resolve dataset path based on base directory and target SNR."""
    if not dir_path or not os.path.exists(dir_path):
        if SOURCE_DIR and os.path.exists(SOURCE_DIR):
            print(f"[Warning] Dataset path '{dir_path}' not found, falling back to '{SOURCE_DIR}'")
            dir_path = SOURCE_DIR
        else:
            return dir_path

    primary_names = ['matlabNTN.mat', 'channel_dur_randomizedUE.mat', 'channel_dur.mat', 'channel_sur.mat', 'channel.mat', 'dataset.mat', 'data.mat']
    excluded_prefixes = ('inferredChannel', 'testChannel', 'training_history', 'extracted_features', 'evaluation_results', 'channel_grids', 'synthesized', 'sample_')

    def find_best_mat_in_dir(folder):
        if not os.path.isdir(folder):
            return None
        for name in primary_names:
            p = os.path.join(folder, name)
            if os.path.isfile(p):
                return p
        for f in sorted(os.listdir(folder)):
            if f.endswith('.mat') and not f.startswith(excluded_prefixes):
                return os.path.join(folder, f)
        return None

    candidates = [
        f"SNR_{snr}dB", f"{snr}dB", f"SNR_{snr}", str(snr),
        f"SNR_{snr:02d}dB", f"snr_{snr}db", f"snr_{snr}"
    ]
    for cand_name in candidates:
        cand_dir = os.path.join(dir_path, cand_name)
        mat = find_best_mat_in_dir(cand_dir)
        if mat:
            return mat

    mat = find_best_mat_in_dir(dir_path)
    if mat:
        return mat

    for root, _, files in os.walk(dir_path):
        for name in primary_names:
            if name in files:
                return os.path.join(root, name)
        for f in sorted(files):
            if f.endswith('.mat') and not f.startswith(excluded_prefixes):
                return os.path.join(root, f)

    return os.path.join(dir_path, 'matlabNTN.mat')


def load_dataset_cgan(mat_filepath: str, input_type: str = 'li') -> dict:
    """
    Load channel data from MATLAB v7.3 (HDF5) or legacy v7 format.
    Returns dictionary with channel arrays formatted into [N, 132, 14] complex.
    """
    input_type = input_type.lower()
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
    mat_dict['H_perfect'] = H_perfect

    H_perfect_ori = None
    for k in ['H_perfect_ori', 'H_perfect_original', 'H_true_ori', 'H_ori']:
        if k in mat_dict and mat_dict[k] is not None:
            H_perfect_ori = format_3d(mat_dict[k])
            break
    mat_dict['H_perfect_ori'] = H_perfect_ori if H_perfect_ori is not None else H_perfect

    if 'pilot_rows' in mat_dict:
        mat_dict['pilot_rows'] = np.squeeze(mat_dict['pilot_rows'])
    if 'pilot_cols' in mat_dict:
        mat_dict['pilot_cols'] = np.squeeze(mat_dict['pilot_cols'])

    input_key_map = {'prac': 'H_prac', 'li': 'H_li', 'ls': 'H_ls_pilots'}
    target_key = input_key_map.get(input_type, f'H_{input_type}')
    if target_key not in mat_dict or mat_dict[target_key] is None:
        for alt in ['H_li', 'H_prac', 'H_ls_pilots', 'H_ls', 'H_perfect']:
            if alt in mat_dict and mat_dict[alt] is not None:
                target_key = alt
                break

    H_in = format_3d(mat_dict[target_key])
    mat_dict[f'H_{input_type}'] = H_in
    mat_dict['H_input'] = H_in

    if 'H_li' in mat_dict:
        mat_dict['H_li'] = format_3d(mat_dict['H_li'])

    return mat_dict


def split_indices(N: int, train_frac: float = 0.70, val_frac: float = 0.15, seed: int = 1234):
    """Return reproducible (train, val, test) index arrays."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    n_train = int(N * train_frac)
    n_val = int(N * val_frac)
    return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]


def compute_ssim_fast(H_pred: np.ndarray, H_true: np.ndarray) -> float:
    """Vectorized SSIM computation across entire batch."""
    real_pred = np.stack([H_pred.real, H_pred.imag], axis=-1).astype(np.float32)
    real_true = np.stack([H_true.real, H_true.imag], axis=-1).astype(np.float32)
    val_range = max(float(np.max(real_true) - np.min(real_true)), 1e-8)
    s = tf_ssim(
        tf.convert_to_tensor(real_true, dtype=tf.float32),
        tf.convert_to_tensor(real_pred, dtype=tf.float32),
        max_val=val_range
    )
    return float(tf.reduce_mean(s).numpy())


def infer_and_evaluate_fast(generator, ds, is_residual=False, lower_range=-1.0):
    """Full-dataset inference and metric evaluation over tf.data.Dataset."""
    all_preds = []
    all_trues = []
    total_se = 0.0
    total_pe = 0.0
    total_samples = 0

    for x_batch, y_batch in ds:
        x_norm, x_min, x_max = batch_minmax_scale(x_batch, lower_range)
        pred_norm, _ = generator(x_norm, training=False)
        pred_real = batch_minmax_descale(pred_norm, x_min, x_max, lower_range).numpy()
        if is_residual:
            pred_real = x_batch.numpy() + pred_real

        pred_c = pred_real[..., 0] + 1j * pred_real[..., 1]
        y_np = y_batch.numpy()
        true_c = y_np[..., 0] + 1j * y_np[..., 1]

        all_preds.append(pred_c)
        all_trues.append(true_c)

        err = true_c - pred_c
        total_se += float(np.sum(np.abs(err)**2))
        total_pe += float(np.sum(np.abs(true_c)**2))
        total_samples += pred_c.shape[0]

    H_pred_all = np.concatenate(all_preds, axis=0) if all_preds else np.empty((0,))
    H_true_all = np.concatenate(all_trues, axis=0) if all_trues else np.empty((0,))

    nmse = float(total_se / (total_pe + 1e-12)) if total_pe > 0 else 0.0
    nmse_db = float(10.0 * np.log10(nmse + 1e-12)) if nmse > 0 else 0.0
    mmse = float(total_se / (total_samples * 132 * 14)) if total_samples > 0 else 0.0
    ssim = compute_ssim_fast(H_pred_all, H_true_all) if len(H_pred_all) > 0 else 0.0

    return H_pred_all, {'nmse': nmse, 'nmse_db': nmse_db, 'mmse': mmse, 'ssim': ssim}


def extract_dataset_features_fast(generator, ds, lower_range=-1.0, selected_layers=None):
    """Extract intermediate representations across dataset without redundant conversions."""
    if not selected_layers:
        return {}
    layer_feats = {lyr: [] for lyr in selected_layers}

    for x_batch, _ in ds:
        x_norm, _, _ = batch_minmax_scale(x_batch, lower_range)
        _, feats = generator(x_norm, training=False, return_features=True)
        for lyr, f_t in zip(selected_layers, feats):
            # Flatten or GAP
            if len(f_t.shape) == 4:
                f_pool = tf.reduce_mean(f_t, axis=[1, 2])
            else:
                f_pool = tf.reshape(f_t, [tf.shape(f_t)[0], -1])
            layer_feats[lyr].append(f_pool.numpy())

    return {lyr: np.concatenate(layer_feats[lyr], axis=0) if layer_feats[lyr] else np.empty((0,)) for lyr in selected_layers}


def save_test_channel_mat(mat_dict, test_indices, H_pred_test, save_filepath, snr_val, model_type_tag):
    """Save test channel grids matching the benchmark schema into testChannel_*.mat."""
    test_dict = {}
    if 'H_perfect' in mat_dict and mat_dict['H_perfect'] is not None:
        test_dict['H_perfect_test'] = mat_dict['H_perfect'][test_indices]

    if 'H_perfect_ori' in mat_dict and mat_dict['H_perfect_ori'] is not None:
        test_dict['H_original_test'] = mat_dict['H_perfect_ori'][test_indices]
    elif 'H_perfect' in mat_dict and mat_dict['H_perfect'] is not None:
        test_dict['H_original_test'] = test_dict['H_perfect_test']

    # 3. H_LS_test (LS pilot sequence or initial estimation grid)
    H_ls_candidate = None
    for k in [f'H_{model_type_tag.lower()}', 'H_input', 'H_ls_pilots', 'H_ls', 'H_li', 'H_prac']:
        if k in mat_dict and mat_dict[k] is not None:
            arr = mat_dict[k]
            if isinstance(arr, np.ndarray) and arr.ndim >= 2:
                n_total = len(mat_dict['H_perfect'])
                if arr.shape[0] == n_total:
                    H_ls_candidate = arr[test_indices]
                elif arr.shape[-1] == n_total:
                    H_ls_candidate = arr[..., test_indices]
                else:
                    H_ls_candidate = arr
                break
    if H_ls_candidate is not None:
        test_dict['H_LS_test'] = H_ls_candidate

    # 4. Pilot coordinates (1-indexed for MATLAB compatibility)
    if 'pilot_rows' in mat_dict and mat_dict['pilot_rows'] is not None:
        p_rows = np.squeeze(mat_dict['pilot_rows'])
        test_dict['pilot_rows'] = p_rows + 1 if np.min(p_rows) == 0 else p_rows
    if 'pilot_cols' in mat_dict and mat_dict['pilot_cols'] is not None:
        p_cols = np.squeeze(mat_dict['pilot_cols'])
        test_dict['pilot_cols'] = p_cols + 1 if np.min(p_cols) == 0 else p_cols

    # 5. Benchmark LI Channel Grid if present
    if 'H_li' in mat_dict and mat_dict['H_li'] is not None:
        arr_li = mat_dict['H_li']
        n_total = len(mat_dict['H_perfect'])
        if isinstance(arr_li, np.ndarray) and arr_li.shape[0] == n_total:
            test_dict['H_LI_test'] = arr_li[test_indices]
        elif isinstance(arr_li, np.ndarray) and arr_li.shape[-1] == n_total:
            test_dict['H_LI_test'] = arr_li[..., test_indices]

    test_dict['H_output_test'] = H_pred_test
    test_dict['test_indices'] = test_indices
    test_dict['snr'] = snr_val
    test_dict['model_type'] = model_type_tag

    savemat(save_filepath, test_dict)
    print(f"[Save] Exported test MAT file -> {save_filepath}")


def plot_loss_curves(history: dict, save_dir: str):
    epochs = range(1, len(history['train_loss']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['train_loss'], label='Total Generator Loss', color='blue', lw=2)
    if 'train_est_loss' in history:
        ax.plot(epochs, history['train_est_loss'], label='Estimation Loss (Source)', color='green', lw=1.5)
    if 'train_domain_loss' in history:
        ax.plot(epochs, history['train_domain_loss'], label='CORAL Loss', color='red', lw=1.5, ls='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('cGAN CORAL Training Loss Progression')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    out_pdf = os.path.join(save_dir, 'loss_total.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)


def plot_val_curves(history: dict, save_dir: str):
    if not history.get('nmse_val_target'):
        return
    epochs = range(1, len(history['nmse_val_target']) + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    if 'nmse_val_source' in history:
        ax.plot(epochs, history['nmse_val_source'], label='Source Val NMSE', color='blue', lw=1.5)
    ax.plot(epochs, history['nmse_val_target'], label='Target Val NMSE', color='orange', lw=2)
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
    """Generate side-by-side visualization PDF of Ground Truth, Input, and Reconstruction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].imshow(np.abs(H_true_sample), aspect='auto', cmap='jet')
    axes[0].set_title("Ground Truth Channel |H|")
    axes[0].set_xlabel("OFDM Symbol")
    axes[0].set_ylabel("Subcarrier")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(np.abs(H_in_sample), aspect='auto', cmap='jet')
    axes[1].set_title("Input Channel |H_in|")
    axes[1].set_xlabel("OFDM Symbol")
    axes[1].set_ylabel("Subcarrier")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(np.abs(H_pred_sample), aspect='auto', cmap='jet')
    axes[2].set_title("cGAN Reconstructed Channel |H_pred|")
    axes[2].set_xlabel("OFDM Symbol")
    axes[2].set_ylabel("Subcarrier")
    plt.colorbar(im2, ax=axes[2])

    fig.tight_layout()
    out_pdf = os.path.join(save_dir, f'{prefix}_sample_reconstruction.pdf')
    fig.savefig(out_pdf)
    plt.close(fig)
    print(f"[Save] Exported channel reconstruction PDF -> {out_pdf}")


# ============================================================================
# MAIN HIGH-PERFORMANCE TRAINING PIPELINE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="High-Performance cGAN Pix2Pix Domain Adaptation with Multi/Single-Layer CORAL for OpenNTN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--source-dir', type=str, default=SOURCE_DIR, help="Source dataset directory containing matlabNTN.mat")
    parser.add_argument('--target-dir', type=str, default=TARGET_DIR, help="Target dataset directory containing matlabNTN.mat")
    parser.add_argument('--save-dir', type=str, default=DEFAULT_SAVE_DIR, help="Folder directory to save results (defaults to './results')")
    parser.add_argument('--snr', type=int, default=DEFAULT_SNR, help="Channel SNR in dB (e.g. -15, -10, -5, 0, 5, 10, 15)")
    parser.add_argument('--type', type=str, default=MODEL_TYPE, choices=['LI', 'LS', 'Prac', 'li', 'ls', 'prac'], help="Model input type")
    parser.add_argument('--only-source', action='store_true', default=ONLY_SOURCE, help="Train using source-only data (no CORAL)")
    parser.add_argument('--residual', action='store_true', default=RESIDUAL, help="Use residual learning for cGAN")
    parser.add_argument(
        '--coral-layers', 
        nargs='+', 
        default=CORAL_LAYERS,
        help="Layers to extract for CORAL loss alignment (e.g., '--coral-layers d3 d4' or '--coral-layers d2 d3 d4' or '--coral-layers d4')"
    )
    parser.add_argument('--train-frac', type=float, default=DEFAULT_TRAIN_FRAC, help="Fraction of data for training")
    parser.add_argument('--val-frac', type=float, default=DEFAULT_VAL_FRAC, help="Fraction of data for validation")
    parser.add_argument('--n-epochs', type=int, default=N_EPOCHS, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument('--test-code', action='store_true', default=TEST_CODE, help="Run with subset of data for testing")
    parser.add_argument('--lower-range', type=float, default=-1.0, choices=[0.0, -1.0], help="Scaling range for minmax ([-1, 1] or [0, 1])")
    parser.add_argument('--adv-weight', type=float, default=0.005, help="GAN adversarial loss weight")
    parser.add_argument('--est-weight', type=float, default=1.0, help="Estimation loss weight")
    parser.add_argument('--domain-weight', type=float, default=0.5, help="CORAL loss weight")
    parser.add_argument('--temporal-weight', type=float, default=0.02, help="Temporal smoothness weight")
    parser.add_argument('--frequency-weight', type=float, default=0.1, help="Frequency smoothness weight")
    parser.add_argument('--save-features', action='store_true', default=SAVE_FEATURES, help="Extract and save intermediate features at begin, mid, and last epoch")
    parser.add_argument('--no-gpu', action='store_true', help="Disable GPU execution")

    args = parser.parse_args()

    if args.no_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        print("[Config] GPU disabled by user.")

    # Parse and validate extracted layers
    selected_layers = []
    for item in args.coral_layers:
        for sub in str(item).replace(',', ' ').split():
            sub_clean = sub.strip().lower()
            if sub_clean and sub_clean not in selected_layers:
                selected_layers.append(sub_clean)

    valid_layers = ['d1', 'd2', 'd3', 'd4', 'u1', 'u2', 'u3']
    for lyr in selected_layers:
        if lyr not in valid_layers:
            raise ValueError(f"Invalid layer '{lyr}' in --coral-layers. Valid options are: {valid_layers}")

    if args.train_frac + args.val_frac >= 1.0:
        raise ValueError(f"train_frac ({args.train_frac}) + val_frac ({args.val_frac}) must be < 1.0 to leave room for test set.")
    test_frac = 1.0 - args.train_frac - args.val_frac

    args.type = args.type.upper()
    domain_weight = 0.0 if args.only_source else args.domain_weight

    # Resolve dataset paths dynamically
    source_data_file_path = os.path.abspath(get_mat_file(args.source_dir, args.snr))
    target_data_file_path = os.path.abspath(get_mat_file(args.target_dir, args.snr))

    print("=" * 80)
    print(f"cGAN High-Performance Domain Adaptation | Mode: {'Source-Only' if args.only_source else 'CORAL UDA'}")
    if not args.only_source:
        print(f"CORAL Extracted Layers: {selected_layers} ({'Multi-layer' if len(selected_layers) > 1 else 'Single-layer'})")
        print(f"CORAL Loss Weight (lambda): {domain_weight}")
    print(f"Architecture: {'Residual cGAN' if args.residual else 'Direct cGAN'}")
    print(f"Source Dataset: {source_data_file_path}")
    print(f"Target Dataset: {target_data_file_path}")
    print(f"SNR: {args.snr} dB | Input Type: {args.type} | Normalization: [{args.lower_range}, 1.0]")
    print(f"Split: {args.train_frac:.0%} Train / {args.val_frac:.0%} Val / {test_frac:.0%} Test")
    print("=" * 80)

    # Save Directory Setup
    if args.save_dir and args.save_dir.strip():
        output_dir = os.path.abspath(args.save_dir)
    else:
        output_dir = os.path.join(current_dir, 'results')

    os.makedirs(output_dir, exist_ok=True)
    print(f"Experiment results will be saved to: {output_dir}")

    # ============ Load Source and Target data ==============
    source_dict = load_dataset_cgan(source_data_file_path, args.type)
    target_dict = load_dataset_cgan(target_data_file_path, args.type)

    H_true_source = source_dict['H_perfect']
    H_in_source = source_dict['H_input']
    N_samp_source = H_true_source.shape[0]

    H_true_target = target_dict['H_perfect']
    H_in_target = target_dict['H_input']
    N_samp_target = H_true_target.shape[0]

    # Perform reproducible 3-Way Split
    if args.test_code:
        indices_train_source = np.arange(0, 64)
        indices_val_source = np.arange(64, 80)
        indices_test_source = np.arange(80, 96)

        indices_train_target = np.arange(0, 64)
        indices_val_target = np.arange(64, 80)
        indices_test_target = np.arange(80, 96)

        args.n_epochs = 5
    else:
        indices_train_src, indices_val_src, indices_test_src = split_indices(
            N_samp_source, args.train_frac, args.val_frac, seed=1234
        )
        indices_train_tgt, indices_val_tgt, indices_test_tgt = split_indices(
            N_samp_target, args.train_frac, args.val_frac, seed=1234
        )

        n_train_src = (len(indices_train_src) // args.batch_size) * args.batch_size
        n_val_src = (len(indices_val_src) // args.batch_size) * args.batch_size
        n_test_src = (len(indices_test_src) // args.batch_size) * args.batch_size

        n_train_tgt = (len(indices_train_tgt) // args.batch_size) * args.batch_size
        n_val_tgt = (len(indices_val_tgt) // args.batch_size) * args.batch_size
        n_test_tgt = (len(indices_test_tgt) // args.batch_size) * args.batch_size

        indices_train_source = indices_train_src[:n_train_src]
        indices_val_source = indices_val_src[:n_val_src]
        indices_test_source = indices_test_src[:n_test_src]

        indices_train_target = indices_train_tgt[:n_train_tgt]
        indices_val_target = indices_val_tgt[:n_val_tgt]
        indices_test_target = indices_test_tgt[:n_test_tgt]

    print(f"Source split -> Train: {len(indices_train_source)} | Val: {len(indices_val_source)} | Test: {len(indices_test_source)}")
    print(f"Target split -> Train: {len(indices_train_target)} | Val: {len(indices_val_target)} | Test: {len(indices_test_target)}")

    # Pre-stack Real/Imag into float32 contiguous arrays [N, 132, 14, 2]
    H_in_src_real = np.stack([np.real(H_in_source), np.imag(H_in_source)], axis=-1).astype(np.float32)
    H_true_src_real = np.stack([np.real(H_true_source), np.imag(H_true_source)], axis=-1).astype(np.float32)

    H_in_tgt_real = np.stack([np.real(H_in_target), np.imag(H_in_target)], axis=-1).astype(np.float32)
    H_true_tgt_real = np.stack([np.real(H_true_target), np.imag(H_true_target)], axis=-1).astype(np.float32)

    # Build High-Throughput tf.data Pipelines
    ds_src_train = tf.data.Dataset.from_tensor_slices((
        H_in_src_real[indices_train_source], H_true_src_real[indices_train_source]
    )).shuffle(1000, seed=42).batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    ds_tgt_train = tf.data.Dataset.from_tensor_slices((
        H_in_tgt_real[indices_train_target], H_true_tgt_real[indices_train_target]
    )).shuffle(1000, seed=42).batch(args.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)

    ds_src_val = tf.data.Dataset.from_tensor_slices((
        H_in_src_real[indices_val_source], H_true_src_real[indices_val_source]
    )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    ds_tgt_val = tf.data.Dataset.from_tensor_slices((
        H_in_tgt_real[indices_val_target], H_true_tgt_real[indices_val_target]
    )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    ds_src_test = tf.data.Dataset.from_tensor_slices((
        H_in_src_real[indices_test_source], H_true_src_real[indices_test_source]
    )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    ds_tgt_test = tf.data.Dataset.from_tensor_slices((
        H_in_tgt_real[indices_test_target], H_true_tgt_real[indices_test_target]
    )).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Initialize Neural Networks
    generator = Pix2PixGenerator(output_channels=2, extract_layers=selected_layers)
    discriminator = PatchGANDiscriminator(disc_l2=1e-5)

    opt_gen = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
    opt_disc = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.5, beta_2=0.9)

    train_metrics = {
        'train_loss': [],
        'train_est_loss': [],
        'train_disc_loss': [],
        'train_domain_loss': [],
        'train_est_loss_target': []
    }

    val_metrics = {
        'nmse_val_source': [],
        'nmse_val_target': []
    }

    saved_features = {}
    mid_epoch = args.n_epochs // 2
    feature_checkpoint_epochs = {
        0: 'begin',
        mid_epoch: 'mid',
        args.n_epochs - 1: 'last'
    }

    start_time = time.perf_counter()
    print(f"\n[Train] Starting GPU-Accelerated cGAN Training for {args.n_epochs} Epochs ...")

    # Main Training Loop
    for epoch in range(args.n_epochs):
        ep_g_loss = 0.0
        ep_est_loss = 0.0
        ep_d_loss = 0.0
        ep_coral_loss = 0.0
        ep_est_tgt = 0.0
        n_batches = 0

        # Save Intermediate Features at Checkpoint Epochs
        if args.save_features and epoch in feature_checkpoint_epochs:
            stage = feature_checkpoint_epochs[epoch]
            src_feats = extract_dataset_features_fast(generator, ds_src_train, args.lower_range, selected_layers)
            tgt_feats = extract_dataset_features_fast(generator, ds_tgt_train, args.lower_range, selected_layers)
            for lyr in selected_layers:
                layer_num = lyr.replace('d', 'l').replace('u', 'u')
                saved_features[f'features_{stage}_{lyr}_src'] = src_feats[lyr]
                saved_features[f'features_{stage}_{lyr}_tgt'] = tgt_feats[lyr]
                saved_features[f'features_{stage}_{layer_num}_src'] = src_feats[lyr]
                saved_features[f'features_{stage}_{layer_num}_tgt'] = tgt_feats[lyr]
                saved_features[f'features_{stage}_{layer_num}'] = tgt_feats[lyr]
                saved_features[f'features_{stage}_{lyr}'] = tgt_feats[lyr]
            print(f"  [Features Saved] Captured intermediate activations at {stage} epoch ({epoch+1}) for layers: {selected_layers}")

        if args.only_source:
            for x_s, y_s in ds_src_train:
                g_l, e_l, d_l, c_l, et_l = train_step_source_only(
                    generator, discriminator, opt_gen, opt_disc,
                    x_s, y_s,
                    adv_weight=args.adv_weight, est_weight=args.est_weight,
                    temporal_weight=args.temporal_weight, frequency_weight=args.frequency_weight,
                    is_residual=args.residual, lower_range=args.lower_range
                )
                ep_g_loss += float(g_l)
                ep_est_loss += float(e_l)
                ep_d_loss += float(d_l)
                ep_coral_loss += float(c_l)
                ep_est_tgt += float(et_l)
                n_batches += 1
        else:
            for (x_s, y_s), (x_t, y_t) in tf.data.Dataset.zip((ds_src_train, ds_tgt_train)):
                g_l, e_l, d_l, c_l, et_l = train_step_coral(
                    generator, discriminator, opt_gen, opt_disc,
                    x_s, y_s, x_t, y_t,
                    adv_weight=args.adv_weight, est_weight=args.est_weight, domain_weight=domain_weight,
                    temporal_weight=args.temporal_weight, frequency_weight=args.frequency_weight,
                    is_residual=args.residual, lower_range=args.lower_range
                )
                ep_g_loss += float(g_l)
                ep_est_loss += float(e_l)
                ep_d_loss += float(d_l)
                ep_coral_loss += float(c_l)
                ep_est_tgt += float(et_l)
                n_batches += 1

        avg_g_loss = ep_g_loss / max(n_batches, 1)
        avg_est_loss = ep_est_loss / max(n_batches, 1)
        avg_d_loss = ep_d_loss / max(n_batches, 1)
        avg_coral_loss = ep_coral_loss / max(n_batches, 1)
        avg_est_tgt = ep_est_tgt / max(n_batches, 1)

        train_metrics['train_loss'].append(avg_g_loss)
        train_metrics['train_est_loss'].append(avg_est_loss)
        train_metrics['train_disc_loss'].append(avg_d_loss)
        train_metrics['train_domain_loss'].append(avg_coral_loss)
        train_metrics['train_est_loss_target'].append(avg_est_tgt)

        # Fast Validation on GPU
        val_se_s, val_pe_s = 0.0, 0.0
        for x_v, y_v in ds_src_val:
            se, pe = val_step_fast(generator, x_v, y_v, is_residual=args.residual, lower_range=args.lower_range)
            val_se_s += float(se)
            val_pe_s += float(pe)
        val_nmse_s_db = 10.0 * np.log10(max(val_se_s / max(val_pe_s, 1e-12), 1e-12))
        val_metrics['nmse_val_source'].append(val_nmse_s_db)

        val_se_t, val_pe_t = 0.0, 0.0
        for x_v, y_v in ds_tgt_val:
            se, pe = val_step_fast(generator, x_v, y_v, is_residual=args.residual, lower_range=args.lower_range)
            val_se_t += float(se)
            val_pe_t += float(pe)
        val_nmse_t_db = 10.0 * np.log10(max(val_se_t / max(val_pe_t, 1e-12), 1e-12))
        val_metrics['nmse_val_target'].append(val_nmse_t_db)

        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == args.n_epochs - 1:
            print(f"Epoch {epoch+1:03d}/{args.n_epochs:03d} | G Loss: {avg_g_loss:.4f} (Est: {avg_est_loss:.4f}, CORAL: {avg_coral_loss:.4f}) | Val Target NMSE: {val_nmse_t_db:.2f} dB")

    total_time = time.perf_counter() - start_time
    print(f"\n[Done] Training completed in {total_time:.2f} seconds ({total_time / args.n_epochs:.3f} s/epoch).")

    # ============================================================================
    # FINAL TEST EVALUATION & testChannel_*.mat EXPORT
    # ============================================================================
    print("\n" + "=" * 80)
    print("                      FINAL TEST PERFORMANCE SUMMARY                  ")
    print("=" * 80)

    # 1. Source Test Evaluation
    H_pred_test_src, metrics_test_src = infer_and_evaluate_fast(
        generator, ds_src_test, is_residual=args.residual, lower_range=args.lower_range
    )
    print(f"  Source Domain -> NMSE: {metrics_test_src['nmse_db']:.2f} dB | MMSE: {metrics_test_src['mmse']:.6e} | SSIM: {metrics_test_src['ssim']:.4f}")

    # 2. Target Test Evaluation
    H_pred_test_tgt, metrics_test_tgt = infer_and_evaluate_fast(
        generator, ds_tgt_test, is_residual=args.residual, lower_range=args.lower_range
    )
    print(f"  Target Domain -> NMSE: {metrics_test_tgt['nmse_db']:.2f} dB | MMSE: {metrics_test_tgt['mmse']:.6e} | SSIM: {metrics_test_tgt['ssim']:.4f}")
    print("=" * 80)

    # 3. Save testChannel_source.mat and testChannel_target.mat
    save_test_channel_mat(
        source_dict, indices_test_source, H_pred_test_src,
        os.path.join(output_dir, 'testChannel_source.mat'),
        args.snr, args.type
    )

    save_test_channel_mat(
        target_dict, indices_test_target, H_pred_test_tgt,
        os.path.join(output_dir, 'testChannel_target.mat'),
        args.snr, args.type
    )

    # 4. Save consolidated final_epoch.txt report matching schema
    txt_path = os.path.join(output_dir, 'final_epoch.txt')
    try:
        with open(txt_path, 'w') as f:
            f.write("=== FINAL EPOCH EVALUATION RESULTS ===\n")
            f.write(f"SNR (dB):             {args.snr}\n")
            f.write(f"Input Type:           {args.type}\n")
            f.write(f"Domain Adaptation:    {'Source-Only' if args.only_source else 'CORAL UDA'}\n")
            f.write(f"CORAL Layers:         {selected_layers}\n")
            f.write(f"Residual Mode:        {args.residual}\n")
            f.write(f"Total Execution Time: {total_time:.1f} s\n\n")

            f.write("--- SOURCE DOMAIN TEST METRICS ---\n")
            f.write(f"Model Output MMSE:    {metrics_test_src['mmse']:e}\n")
            f.write(f"Model Output NMSE:    {metrics_test_src['nmse']:e} ({metrics_test_src['nmse_db']:.2f} dB)\n")
            f.write(f"Model Output SSIM:    {metrics_test_src['ssim']:.4f}\n\n")

            f.write("--- TARGET DOMAIN TEST METRICS ---\n")
            f.write(f"Model Output MMSE:    {metrics_test_tgt['mmse']:e}\n")
            f.write(f"Model Output NMSE:    {metrics_test_tgt['nmse']:e} ({metrics_test_tgt['nmse_db']:.2f} dB)\n")
            f.write(f"Model Output SSIM:    {metrics_test_tgt['ssim']:.4f}\n")
        print(f"[Save] Final epoch text report -> {txt_path}")
    except Exception as e:
        print(f"[Save Warning] Failed to write final_epoch.txt: {e}")

    # 5. Save consolidated evaluation_results.mat
    eval_dict = {
        'nmse_test_src': metrics_test_src['nmse'],
        'nmse_test_src_db': metrics_test_src['nmse_db'],
        'mmse_test_src': metrics_test_src['mmse'],
        'ssim_test_src': metrics_test_src['ssim'],
        'nmse_test_tgt': metrics_test_tgt['nmse'],
        'nmse_test_tgt_db': metrics_test_tgt['nmse_db'],
        'mmse_test_tgt': metrics_test_tgt['mmse'],
        'ssim_test_tgt': metrics_test_tgt['ssim'],
        'mmse_test': metrics_test_tgt['mmse'],
        'nmse_test': metrics_test_tgt['nmse'],
        'nmse_test_db': metrics_test_tgt['nmse_db'],
        'ssim_test': metrics_test_tgt['ssim'],
        'indices_train_source': indices_train_source,
        'indices_val_source': indices_val_source,
        'indices_test_source': indices_test_source,
        'indices_train_target': indices_train_target,
        'indices_val_target': indices_val_target,
        'indices_test_target': indices_test_target,
        'snr': args.snr,
        'input_type': args.type,
        'residual': args.residual,
        'coral_layers': np.array(selected_layers)
    }
    eval_path = os.path.join(output_dir, 'evaluation_results.mat')
    savemat(eval_path, eval_dict)
    print(f"[Save] Evaluation results -> {eval_path}")

    # 6. Save History MAT
    history = {}
    history.update(train_metrics)
    history.update(val_metrics)
    history_save_path = os.path.join(output_dir, 'training_history.mat')
    savemat(history_save_path, {k: np.array(v) for k, v in history.items()})
    print(f"[Save] Saved training history -> {history_save_path}")

    # 7. Save Extracted Features MAT if requested
    if args.save_features and saved_features:
        feat_save_path = os.path.join(output_dir, 'extracted_features.mat')
        saved_features['selected_layers'] = np.array(selected_layers)
        saved_features['train_indices_src'] = indices_train_source
        saved_features['train_indices_tgt'] = indices_train_target
        savemat(feat_save_path, saved_features, do_compression=True)
        print(f"[Save] Exported extracted features -> {feat_save_path}")

    # 8. Save Visualizations
    try:
        plot_loss_curves(train_metrics, output_dir)
        plot_val_curves(val_metrics, output_dir)
        if len(H_pred_test_tgt) > 0:
            sample_true = H_true_target[indices_test_target[0]]
            sample_in = H_in_target[indices_test_target[0]]
            save_channel_plots_pdf(sample_true, sample_in, H_pred_test_tgt[0], output_dir, prefix='target_test')
    except Exception as e:
        print(f"[Plot Warning] Failed to render PDF plots: {e}")

    print(f"\n[Done] High-performance cGAN training and evaluation complete. Results saved in: {output_dir}")


if __name__ == '__main__':
    main()
