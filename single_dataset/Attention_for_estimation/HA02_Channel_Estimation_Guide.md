Viewed ha02_channel_estimation.py:22-45

Here is the step-by-step trace of how the data flows through the HA02 model, which clarifies exactly where the attention is and the correct order of the CNN vs. fully connected layers:

```
Inputs (B, 88, 2)
  │
  ▼
┌────────────────────────────────────────┐
│  1. TransformerEncoderBlock            │  <--- THIS IS WHERE SELF-ATTENTION HAPPENS!
│  - Multi-Head Self-Attention on pilots │
│  - Learns global correlations between  │
│    the 88 pilot points                 │
└────────────────────────────────────────┘
  │
  ▼
Output remains (B, 88, 2)
  │
  ▼
┌────────────────────────────────────────┐
│  2. Format for CNN                     │
│  - Expands to (B, 88, 2, 1)            │
└────────────────────────────────────────┘
  │
  ▼
┌────────────────────────────────────────┐
│  3. Conv2D + Residual Conv2D Blocks    │  <--- CNN IS APPLIED ON PILOTS FIRST!
│  - Processes spatial structures in     │
│    the sparse pilot dimension          │
└────────────────────────────────────────┘
  │
  ▼
Shape is (B, 88, 2, n_filter)
  │
  ▼
┌────────────────────────────────────────┐
│  4. Fully Connected (FC) Upsampling   │  <--- UPSAMPLING / PROJECTION
│  - self.fc_upsample projects the       │
│    last dimension (88) to 1848 (14*132)│
└────────────────────────────────────────┘
  │
  ▼
Shape is (B, 1848, 2, n_filter)
  │
  ▼
┌────────────────────────────────────────┐
│  5. Final Conv2D                       │
│  - Maps n_filter channels to 1 channel │
└────────────────────────────────────────┘
  │
  ▼
Shape is (B, 1848, 2, 1)
  │
  ▼
┌────────────────────────────────────────┐
│  6. Final Reshape                      │
│  - Squeezes & reshapes to final grid   │
└────────────────────────────────────────┘
  │
  ▼
Outputs (B, 14, 132, 2)
```

---

### Detailed Answers:

1. **Where is the Attention?**
   The attention happens in the very first block: the **`TransformerEncoderBlock`**. It treats the $88$ sparse pilot points as a sequence of length $88$ (with $2$ features each: real and imaginary). It runs multi-head self-attention on this sequence to learn how each pilot's value relates to every other pilot's value before any interpolation or upsampling begins. The output shape remains `(B, 88, 2)`.

2. **Is it all same-shape CNN layers after upsampling?**
   Actually, the order is reversed: **the CNN layers are run BEFORE the upsampling**. 
   * First, `conv1` and the residual CNN layers process the raw $88 \times 2$ pilot representation.
   * Then, the fully connected layer (`fc_upsample`) expands the spatial pilot dimension from $88$ to $1848$ ($14 \times 132$).
   * Finally, one last Conv2D (`conv_out`) is applied to map the features back to the target grid structure, which is then reshaped to `(B, 14, 132, 2)`.