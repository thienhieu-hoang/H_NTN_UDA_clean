Looking directly into your [`run_LS_Attention_CORAL.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/run_LS_Attention_CORAL.py) and [`run_LS_Attention_pHeadCORAL.py`](file:///c:/Users/AT30890/Hoctap/1_Hprediction/working/H_predict_NTN/Hest_NTN_UDA_clean/CORAL/run_LS_Attention_pHeadCORAL.py), your current implementation treats the network like a generic feed-forward MLP/CNN rather than a Transformer.

---

### 1. Diagnosis of the Current CORAL Implementation

Here is what your code currently does:
```python
# In TransformerEncoderBlock:
z_enc = self.ln2(x_norm1 + ffn_out)  # Shape: (B, 176)

# In compute_covariance(features):
# features has shape (B, 176)
cov_matrix = (Z - mean)^T (Z - mean) / (B - 1)  # Shape: (176, 176)
```

#### Why this is "naive" for a Transformer:
1. **Flattens Token Structure:** It crushes $88\text{ pilot tokens} \times 2\text{ channels (Real, Imag)}$ into a flat 176-D vector. The model forgets that token $i$ corresponds to a specific pilot subcarrier and symbol location.
2. **Computes Covariance Only Across Batch $B$:** With a batch size of, say, $B = 32$ or $64$, estimating a $176 \times 176$ covariance matrix is statistically ill-conditioned ($B < d$).
3. **Completely Ignores the Attention Mechanism:** The core of a Transformer is the **Attention Map** $\mathbf{A} \in \mathbb{R}^{B \times 2 \times 88 \times 88}$. The current CORAL loss never looks at the attention weights or the Query/Key projections.

---

### 2. Physical Insight: What Should Be Aligned in a Wireless Transformer?

In wireless channel estimation, what does the attention map actually represent?
$$\mathbf{Scores}_{i,j} = \frac{\mathbf{q}_i \mathbf{k}_j^T}{\sqrt{d_k}}, \quad \mathbf{A}_{i,j} = \text{Softmax}(\mathbf{Scores}_{i,j})$$

- $\mathbf{A}_{i,j}$ represents **how much pilot $i$ should attend to pilot $j$** to interpolate the channel.
- In signal processing, this is the neural network's internal approximation of the **pilot correlation matrix**:
  $$\mathbf{R}_{pp}[i, j] = \mathbb{E}[h(f_i, t_i) h^*(f_j, t_j)]$$
- When the domain shifts (e.g., target domain has a higher Doppler shift or larger delay spread), the true physical correlation between pilot $i$ and pilot $j$ changes.
- **The problem:** A source-trained Transformer applies the **source domain's correlation pattern** to target pilot tokens, blending pilots that are no longer correlated!

---

### 3. Three Recommended Ways to Implement Transformer-Native CORAL

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        3 TRANSFORMER-NATIVE ALIGNMENT STRATEGIES                       │
├────────────────────────────┬─────────────────────────────┬─────────────────────────────┤
│ Strategy A: Attention Map  │ Strategy B: Query-Key (Q-K) │ Strategy C: Token-Pool      │
│ Alignment (Attn-CORAL)     │ Metric Space Alignment      │ Covariance Matching         │
│ • Directly aligns pilot-to-│ • Aligns Q and K projections│ • Treats 88 pilots as       │
│   pilot correlation A_ij   │ • Ensures valid dot-products│   tokens, pools B x 88      │
└────────────────────────────┴─────────────────────────────┴─────────────────────────────┘
```

---

#### 🌟 Strategy A (Highest Physical Relevance): Attention Map Alignment
Instead of aligning feature activations, align the **mean attention maps** between source and target!

In your `TransformerEncoderBlock`, expose `attn_weights` (shape: `[B, num_heads=2, 88, 88]`):
```python
# 1. Compute the average attention map across the batch for source and target
# Shape: [num_heads, 88, 88]
mean_attn_src = tf.reduce_mean(attn_weights_src, axis=0)
mean_attn_tgt = tf.reduce_mean(attn_weights_tgt, axis=0)

# 2. Minimize Frobenius discrepancy between learned pilot correlations:
loss_attn = tf.reduce_mean(tf.square(mean_attn_src - mean_attn_tgt))
```
* **Why this is powerful:** It directly forces the attention heads to adjust their subcarrier-weighting policy to match the target channel's physical coherence bandwidth and coherence time.

---

#### 🌟 Strategy B (Most Elegant for Q-K): Query-Key Subspace Alignment
The attention weights are computed from queries $\mathbf{Q}$ and keys $\mathbf{K}$:
```python
Q = qkv[:, 0, :, :]  # (B, 2, 88)
K = qkv[:, 1, :, :]  # (B, 2, 88)
```
If the feature distributions of $\mathbf{Q}$ and $\mathbf{K}$ diverge between domains, the dot-product $\mathbf{Q} \mathbf{K}^T$ loses its geometric meaning.

Align the covariance of $\mathbf{Q}$ and $\mathbf{K}$ directly:
```python
# Reshape Q and K to [B, 2 * 88 = 176]
Q_flat_src = tf.reshape(Q_src, [B, -1])
Q_flat_tgt = tf.reshape(Q_tgt, [B, -1])

K_flat_src = tf.reshape(K_src, [B, -1])
K_flat_tgt = tf.reshape(K_tgt, [B, -1])

# CORAL on Query and Key spaces
loss_coral_Q = coral_loss_single_layer(Q_flat_src, Q_flat_tgt)
loss_coral_K = coral_loss_single_layer(K_flat_src, K_flat_tgt)
loss_QK = loss_coral_Q + loss_coral_K
```
* **Why this works:** It preserves the relative angles between pilot queries and keys so that self-attention remains well-scaled in the target domain.

---

#### 🌟 Strategy C (Best Statistical Stability): Token-Pooled Covariance
Currently, you compute covariance across batch $B$ on length-176 vectors.
Instead, view your channel as **$88$ tokens of feature dimension $d = 2$** (or $d = 64$ with a projection head):

```python
# In TransformerEncoderBlock:
# Instead of z_enc as [B, 176], keep tokens as [B, 88, 2]
tokens_src = tf.reshape(z_enc_src, [-1, 2])  # Shape: [B * 88, 2]
tokens_tgt = tf.reshape(z_enc_tgt, [-1, 2])  # Shape: [B * 88, 2]

# Compute 2x2 covariance across ALL B * 88 tokens:
cov_src = compute_covariance(tokens_src)  # [2, 2]
cov_tgt = compute_covariance(tokens_tgt)  # [2, 2]

loss_token_coral = tf.reduce_sum(tf.square(cov_src - cov_tgt))
```
* **Why this is better statistically:** Instead of estimating a $176 \times 176$ matrix from only $B=32$ samples (which is rank-deficient), you estimate a compact matrix from **$32 \times 88 = 2,816$ samples**, making the covariance estimate rock-solid.

---

### Comparison of the 3 Strategies

| Strategy | What it Aligns | Mathematical Shape | Key Benefit |
| :--- | :--- | :---: | :--- |
| **Current Code** | Flattened post-FFN vector | $[B, 176]$ | Baseline (Treats model as MLP) |
| **Strategy A (Attn Map)** | Attention weights $\mathbf{A}$ | $[B, 2, 88, 88]$ | Directly adapts **pilot correlation physics** |
| **Strategy B (Q-K Space)** | Query & Key projections | $[B, 176]$ | Stabilizes dot-product metric space |
| **Strategy C (Token Pool)** | Pilot tokens pooled across space | $[B \times 88, d]$ | **Rock-solid covariance estimate** ($N \gg d$) |

---

### Recommended Implementation Step

If you want to upgrade your `LS + Transformer` UDA:
1. **Combine Strategy A + Strategy C**:
   - Use **Strategy A** on the attention weights: $\mathcal{L}_{\text{attn}}$ ensures the pilot-to-pilot attention matrix adapts to the target delay/Doppler spread.
   - Use **Strategy C** with your `ProjectionHead`: project the 88 pilot tokens into a 64-D space, pool them across $B \times 88$, and run CORAL.
2. This replaces the naive flattening with a physically grounded, mathematically well-conditioned Transformer adaptation pipeline.