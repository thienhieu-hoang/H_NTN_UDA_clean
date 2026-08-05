Yes, your `CNNGenerator` uses **ResNet blocks (`SameShapeBlock`) for all of its internal blocks**.

Since your network is a 2D Same-Shape Refining Network, here is how you can **add Attention / Transformer modules without changing the same-shape structure or sequence format**.

---

### How Attention Works on a 2D Grid

Even though your feature map is a 2D matrix of shape **$(B, 132, 14, C)$**, you can easily apply Attention by flattening the 2D spatial dimensions into a sequence of $132 \times 14 = 1848$ tokens:

$$\text{2D Feature Map } (B, 132, 14, C) \quad \xrightarrow{\text{Reshape}} \quad \text{Sequence } (B, 1848, C) \quad \xrightarrow{\text{Self-Attention}} \quad \text{Reshape} \quad (B, 132, 14, C)$$

---

### 3 Best Ways to Improve Your Same-Shape ResNet with Attention

#### Option 1: CBAM (Convolutional Block Attention Module) — *Easiest & Most Effective*

CBAM adds two lightweight attention mechanisms to your existing `SameShapeBlock`:
1. **Channel Attention:** Learns *which feature channels* are most important for channel estimation.
2. **Spatial Attention:** Learns *which subcarriers and symbols* in the $132 \times 14$ grid need the most noise correction.

```
SameShapeBlock Output ──► Channel Attention ──► Spatial Attention ──► Refined Feature Map
```

##### Code snippet to add CBAM inside `SameShapeBlock`:

```python
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(1, (7, 7), padding='same', activation='sigmoid')

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_out, max_out], axis=-1)
        scale = self.conv(concat)
        return x * scale  # Scale spatial grid
```

- **Why it's great:** Zero memory explosion, maintains 100% same-shape `(132, 14, C)`, super fast!

---

#### Option 2: Mid-Level Transformer Block (*Restormer / Swin Style*)

Insert **ONE Transformer Block in the middle** of your `CNNGenerator` (e.g. after Block 2, before Block 3):

```
Input (132x14x2)
   │
[ Block 1 (SameShapeBlock) ]
   │
[ Block 2 (SameShapeBlock) ]  --> Local 2D Conv Features (132x14xC)
   │
   ▼
[ Mid-Transformer Block ]     --> Reshape (1848, C) -> Multi-Head Attention -> Reshape (132x14xC)
   │                             (Learns global fading correlations across distant subcarriers)
   ▼
[ Block 3 (SameShapeBlock) ]
   │
[ Block 4 (SameShapeBlock) ]
   │
Output ΔH + Input
```

##### Keras Code for Mid-Level Spatial Transformer Block:

```python
class MidSpatialTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        embed_dim = input_shape[-1]
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads, key_dim=embed_dim // self.num_heads)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ffn = tf.keras.sequential([
            tf.keras.layers.Dense(embed_dim * 2, activation='gelu'),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, x):
        B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
        
        # 1. Flatten 2D grid to sequence: (B, 132x14, C) = (B, 1848, C)
        x_flat = tf.reshape(x, [B, H * W, C])
        
        # 2. Multi-Head Self-Attention + Residual
        attn_out = self.mha(self.ln1(x_flat), self.ln1(x_flat))
        x_norm1 = x_flat + attn_out
        
        # 3. FFN + Residual
        x_out = x_norm1 + self.ffn(self.ln2(x_norm1))
        
        # 4. Reshape back to 2D grid: (B, 132, 14, C)
        return tf.reshape(x_out, [B, H, W, C])
```

---

#### Option 3: Axial / 2D Separated Attention (Subcarrier Attn + Symbol Attn)

Instead of attending to all $1848$ points at once, separate attention along the 2 physical axes:
1. **Subcarrier Attention (Frequency Axis = 132):** Models frequency-selective fading.
2. **Symbol Attention (Time Axis = 14):** Models Doppler shifts over time.

- **Why it's brilliant:** It mimics the physical channel characteristics (Doppler along time, Delay Spread along frequency) while being extremely fast and low-memory!

---

### Summary Recommendation

| Option | Architecture Change | What it improves | Complexity |
|---|---|---|---|
| **1. CBAM** | Add Spatial Attention inside `SameShapeBlock` | Denoising subcarrier spikes | Low (Recommended first step) |
| **2. Mid-Transformer** | Insert 1 Transformer block after Block 2 | Global correlations across grid | Medium |
| **3. Axial Attention** | Separate Frequency & Time Attention | Doppler & Delay Spread modeling | Medium |

Starting with **Option 1 (CBAM)** or **Option 2 (Mid-Transformer)** is the cleanest way to upgrade your Same-Shape ResNet model!