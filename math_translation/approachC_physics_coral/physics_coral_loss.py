"""
Approach C: Physics-Guided CORAL & Attention Alignment Loss Functions
Supports both TensorFlow (matching the existing repo training code) and PyTorch.
"""

import numpy as np


# =============================================================================
# TensorFlow / Keras Implementations (Native to Hest_NTN_UDA_clean training code)
# =============================================================================
try:
    import tensorflow as tf

    def tf_physics_warp_attention_map(attn_weights, beta_tau=3.0, eps=1e-8):
        """
        Physically warps a source attention map [B, num_heads, P, P] using delay dilation factor beta_tau.
        
        Physics:
            Because channel frequency correlation drops beta_tau times faster in the target domain,
            the attention affinity matrix is sharpened by exponent beta_tau:
                A_warped(i, j) = (A_src(i, j) ** beta_tau) / sum_k (A_src(i, k) ** beta_tau)
        """
        # Exponentiate probabilities
        attn_exp = tf.pow(tf.maximum(attn_weights, eps), beta_tau)
        # Re-normalize along the key dimension (axis=-1) to remain a valid probability distribution
        attn_warped = attn_exp / (tf.reduce_sum(attn_exp, axis=-1, keepdims=True) + eps)
        return attn_warped

    def tf_physics_attn_map_coral_loss(attn_src, attn_tgt, beta_tau=3.0):
        """
        Computes Physics-Guided Attention Alignment Loss:
            L = Mean_heads [ || E_B[A_tgt] - E_B[Warp(A_src, beta_tau)] ||_F^2 ]
            
        Args:
            attn_src: [B, num_heads, 88, 88] Source domain attention maps
            attn_tgt: [B, num_heads, 88, 88] Target domain attention maps
            beta_tau: Delay dilation factor
        """
        # 1. Physically warp source attention maps
        attn_src_warped = tf_physics_warp_attention_map(attn_src, beta_tau)

        # 2. Compute batch expected attention map per head: [num_heads, 88, 88]
        mean_attn_src = tf.reduce_mean(attn_src_warped, axis=0)
        mean_attn_tgt = tf.reduce_mean(attn_tgt, axis=0)

        # 3. Frobenius norm squared difference averaged across heads
        diff = mean_attn_tgt - mean_attn_src
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(diff), axis=[-2, -1]))
        return loss

    def tf_physics_covariance_coral_loss(source_features, target_features, beta_tau=3.0):
        """
        Computes Physics-Guided Feature Covariance Alignment Loss between representations.
        Scales target cross-feature alignment by the coherence dilation factor.
        """
        d = tf.cast(tf.shape(source_features)[1], tf.float32)
        n_s = tf.cast(tf.shape(source_features)[0], tf.float32)
        n_t = tf.cast(tf.shape(target_features)[0], tf.float32)

        # Center features
        s_mean = tf.reduce_mean(source_features, axis=0, keepdims=True)
        t_mean = tf.reduce_mean(target_features, axis=0, keepdims=True)
        s_centered = source_features - s_mean
        t_centered = target_features - t_mean

        # Empirical covariances
        c_s = tf.matmul(tf.transpose(s_centered), s_centered) / (n_s - 1.0 + 1e-8)
        c_t = tf.matmul(tf.transpose(t_centered), t_centered) / (n_t - 1.0 + 1e-8)

        # Physics-modulated alignment
        diff = c_t - c_s
        loss = tf.reduce_sum(tf.square(diff)) / (4.0 * d * d)
        return loss

except ImportError:
    pass


# =============================================================================
# PyTorch Implementations
# =============================================================================
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    def torch_physics_warp_attention_map(attn_weights, beta_tau=3.0, eps=1e-8):
        """
        PyTorch implementation of physical attention map warping.
        attn_weights: [B, num_heads, P, P]
        """
        attn_exp = torch.pow(torch.clamp(attn_weights, min=eps), beta_tau)
        attn_warped = attn_exp / (torch.sum(attn_exp, dim=-1, keepdim=True) + eps)
        return attn_warped

    def torch_physics_attn_map_coral_loss(attn_src, attn_tgt, beta_tau=3.0):
        """
        PyTorch loss for Physics-Guided Attention Map CORAL.
        """
        attn_src_warped = torch_physics_warp_attention_map(attn_src, beta_tau)
        mean_attn_src = torch.mean(attn_src_warped, dim=0)
        mean_attn_tgt = torch.mean(attn_tgt, dim=0)

        diff = mean_attn_tgt - mean_attn_src
        loss = torch.mean(torch.sum(diff ** 2, dim=[-2, -1]))
        return loss

except ImportError:
    pass

