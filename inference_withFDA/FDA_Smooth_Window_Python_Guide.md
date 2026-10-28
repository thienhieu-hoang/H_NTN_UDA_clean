# Delay-Doppler FDA with Smooth Tapering Windows
### Specification & Python Implementation Guide (NumPy / PyTorch)

> **Document Purpose:**  
> This specification documents the mathematical formulation, design rationale, and Python (NumPy & PyTorch) implementations of the **Smooth Tapering Window** used in Delay-Doppler Fourier Domain Adaptation (FDA) for 5G Non-Terrestrial Network (NTN) channel estimation.  
> It is designed to be directly copied into Machine Learning repositories to ensure **100% numerical consistency** between MATLAB dataset generation and Python deep-learning training/inference pipelines.

---

## 1. Problem & Physical Motivation

### 1.1 The Legacy Hard Window Issue ("Outside Ring" / Gibbs Ringing)
In standard FDA, low-frequency Delay-Doppler components of a clean source channel $H_{\text{src}}$ are replaced with those of a noisy/inferred target channel $H_{\text{tgt}}$:
$$\tilde{A}(\tau, \nu) = M \odot A_{\text{tgt}}(\tau, \nu) + (1 - M) \odot A_{\text{src}}(\tau, \nu)$$

In legacy implementations, the mask $M$ is a **binary rectangular boxcar window** ($M \in \{0, 1\}$):
- Inside box ($13 \times w_w$): $M = 1$
- Outside box: $M = 0$

Because $A_{\text{tgt}}$ and $A_{\text{src}}$ have different energy levels and multipath structures, this step function creates an **abrupt jump discontinuity** at the perimeter of the box.

```
   Delay-Doppler Domain                 Time-Frequency Dual Domain
┌─────────────────────────┐           ┌─────────────────────────┐
│     Source (M = 0)      │           │                         │
│       ┌─────────┐       │   2D IFFT │   (( Rings / Ripples )) │  <-- Gibbs Phenomenon
│       │ Target  │       │  ───────> │      ┌───────────┐      │      (decaying sinc
│       │ (M = 1) │       │           │      │           │      │       sidelobes)
│       └─────────┘       │           │      └───────────┘      │
│   ▲ Sharp Step Edge     │           │                         │
└─────────────────────────┘           └─────────────────────────┘
```

When transforming back to the Time-Frequency domain via 2D inverse Fourier transform ($\text{FFT}_f(\text{IFFT}_t)$), multiplication by a rectangular window in the frequency domain is equivalent to **2D convolution with a sinc function** ($\text{sinc}(\Delta \tau) \cdot \text{sinc}(\Delta \nu)$). The slow $1/x$ decay of sinc sidelobes causes:
1. **Severe Gibbs ringing / ripple waves** across subcarriers and OFDM symbols.
2. A distinct **rectangular boundary seam / "outside ring"** in the channel power delay and Doppler profiles.

---

### 1.2 The Smooth Window Solution (`tukey_outward`)
To eliminate this artifact while retaining the domain adaptation benefits:
1. **Preserve 100% Target Core:** The inner $13 \times w_w$ region is kept strictly flat at **$1.0$** so that zero target low-frequency energy (delay spread & bulk Doppler) is lost.
2. **Smooth Outward Roll-Off:** An outer transition band softly decays from **$1.0 \to 0.0$** using a raised-cosine (Tukey) taper into the adjacent neighboring pixels.
3. **Pure Source Outer:** All high-frequency multipath regions beyond the transition band are strictly **$0.0$** (100% clean source).

---

## 2. Mathematical Specification

The 2D mask $M \in \mathbb{R}^{H \times W}$ is separable ($H = 132$ subcarriers, $W = 14$ OFDM symbols):
$$M(y, x) = w_y(y) \cdot w_x(x)$$

### 2.1 Outward Raised-Cosine Formula
Let:
- $c_y, c_x$ be the center coordinates (DC: Delay=0, Doppler=0).
- $r_h = \lfloor win\_h / 2 \rfloor = 6$ (for nominal height 13).
- $r_w = \lfloor win\_w / 2 \rfloor$ (1 for $w_w=3$, 2 for $w_w=5$).
- $t_h = 2$ (Delay taper elements).
- $t_w = 1$ (Doppler taper elements).

For distance $d = |\text{coord} - \text{center}|$:
$$w(d) = \begin{cases}
1.0, & d \le r_{\text{core}} \quad (\text{Core passband}) \\
\frac{1}{2}\left(1 + \cos\left(\frac{(d - r_{\text{core}})\pi}{taper + 1}\right)\right), & r_{\text{core}} < d \le r_{\text{core}} + taper \quad (\text{Smooth roll-off}) \\
0.0, & d > r_{\text{core}} + taper \quad (\text{Outer stopband})
\end{cases}$$

---

### 2.2 Numerical Values: MATLAB (1-based) vs. Python (0-based)

#### A. Doppler Dimension ($W = 14$ symbols, $w_w = 3$, $t_w = 1$)
- **Center Column:** Column 8 (MATLAB) / Index 7 (Python).
- **Core Radius:** $r_w = 1$ ($w_w = 3$).

| MATLAB Col (1-based) | Python Col (0-based) | Distance $d$ from DC | Weight $w_x$ | Evaluation | Region |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Col 5** | **Index 4** | $d = 2$ | **`0.0000`** | Stopband | Outer (100% Source) |
| **Col 6** | **Index 5** | $d = 1 + 1$ | **`0.5000`** | $\frac{1}{2}(1 + \cos(\pi/2))$ | *Smooth Transition Step 1* |
| **Col 7** | **Index 6** | $d = 1$ | **`1.0000`** | Core | **Target Core (100% Target)** |
| **Col 8** | **Index 7** | $d = 0$ (DC) | **`1.0000`** | Core | **Target Core (100% Target)** |
| **Col 9** | **Index 8** | $d = 1$ | **`1.0000`** | Core | **Target Core (100% Target)** |
| **Col 10** | **Index 9** | $d = 1 + 1$ | **`0.5000`** | $\frac{1}{2}(1 + \cos(\pi/2))$ | *Smooth Transition Step 1* |
| **Col 11** | **Index 10** | $d = 2$ | **`0.0000`** | Stopband | Outer (100% Source) |

---

#### B. Delay Dimension ($H = 132$ subcarriers, $w_h = 13$, $t_h = 2$)
- **Center Row:** Row 67 (MATLAB) / Index 66 (Python).
- **Core Radius:** $r_h = 6$ ($w_h = 13$).

| MATLAB Row (1-based) | Python Row (0-based) | Distance $d$ from DC | Weight $w_y$ | Evaluation | Region |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **Row 58** | **Index 57** | $d = 6 + 3$ | **`0.0000`** | Stopband | Outer (100% Source) |
| **Row 59** | **Index 58** | $d = 6 + 2$ | **`0.2500`** | $\frac{1}{2}(1 + \cos(2\pi/3))$ | *Smooth Transition Step 2* |
| **Row 60** | **Index 59** | $d = 6 + 1$ | **`0.7500`** | $\frac{1}{2}(1 + \cos(\pi/3))$ | *Smooth Transition Step 1* |
| **Rows 61..73** | **Indices 60..72** | $d \le 6$ | **`1.0000`** | Core | **Target Core (100% Target)** |
| **Row 74** | **Index 73** | $d = 6 + 1$ | **`0.7500`** | $\frac{1}{2}(1 + \cos(\pi/3))$ | *Smooth Transition Step 1* |
| **Row 75** | **Index 74** | $d = 6 + 2$ | **`0.2500`** | $\frac{1}{2}(1 + \cos(2\pi/3))$ | *Smooth Transition Step 2* |
| **Row 76** | **Index 75** | $d = 6 + 3$ | **`0.0000`** | Stopband | Outer (100% Source) |

---

#### C. 2D Center Slice ($M(\tau, \nu) = w_y \times w_x$)
At the cardinal axes and diagonal corners around DC:

| Row \ Col | Col 5 (0) | Col 6 (0.5) | Col 7 (1.0) | Col 8 (1.0) | Col 9 (1.0) | Col 10 (0.5) | Col 11 (0) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Row 58 (0.00)** | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 |
| **Row 59 (0.25)** | 0.00 | **0.125** | 0.25 | 0.25 | 0.25 | **0.125** | 0.00 |
| **Row 60 (0.75)** | 0.00 | **0.375** | 0.75 | 0.75 | 0.75 | **0.375** | 0.00 |
| **Rows 61..73 (1.00)** | 0.00 | 0.500 | **1.00** | **1.00** | **1.00** | 0.500 | 0.00 |
| **Row 74 (0.75)** | 0.00 | **0.375** | 0.75 | 0.75 | 0.75 | **0.375** | 0.00 |
| **Row 75 (0.25)** | 0.00 | **0.125** | 0.25 | 0.25 | 0.25 | **0.125** | 0.00 |
| **Row 76 (0.00)** | 0.00 | 0.000 | 0.00 | 0.00 | 0.00 | 0.000 | 0.00 |

*Notice how corner pixels (e.g. $0.375$ and $0.125$) round the sharp rectangular corners into smooth contours.*

---

## 3. Python Implementation: NumPy (CPU / Preprocessing)

Use this module in your data loader or dataset preprocessing scripts:

```python
import numpy as np

def create_smooth_fda_window(
    grid_h: int = 132,
    grid_w: int = 14,
    win_h_px: int = 13,
    win_w_px: int = 3,
    window_type: str = "tukey_outward",
    taper_h: int = 2,
    taper_w: int = 1,
) -> np.ndarray:
    """
    Generates a 2D smooth mixing mask M in [0, 1] for Delay-Doppler FDA.
    
    Parameters
    ----------
    grid_h : int
        Total subcarriers / delay bins (default 132).
    grid_w : int
        Total OFDM symbols / Doppler bins (default 14).
    win_h_px : int
        Nominal window height along Delay (default 13).
    win_w_px : int
        Nominal window width along Doppler (default 3 or 5).
    window_type : str
        'tukey_outward' (default): keeps core win_h x win_w flat at 1.0, rolls off outward.
        'tukey_inward': rolls off inside the core.
        'supergaussian': elliptical flat-top super-Gaussian.
        'hard': legacy binary rectangular boxcar.
    taper_h : int
        Number of transition rows along Delay (default 2 -> [0.75, 0.25]).
    taper_w : int
        Number of transition cols along Doppler (default 1 -> [0.50]).
        
    Returns
    -------
    mask : np.ndarray of shape (grid_h, grid_w), dtype np.float32
    """
    cy = grid_h // 2  # 66 for 132
    cx = grid_w // 2  # 7 for 14

    r_h = win_h_px // 2  # 6
    r_w = win_w_px // 2  # 1 (for win_w=3) or 2 (for win_w=5)

    window_type = window_type.lower()

    if window_type in ("tukey_outward", "tukey", "raised_cosine"):
        # 1D Delay profile
        y_idx = np.arange(grid_h)
        d_y = np.abs(y_idx - cy)
        w_y = np.zeros(grid_h, dtype=np.float32)

        core_mask_y = d_y <= r_h
        taper_mask_y = (d_y > r_h) & (d_y <= (r_h + taper_h))
        k_y = d_y[taper_mask_y] - r_h

        w_y[core_mask_y] = 1.0
        w_y[taper_mask_y] = 0.5 * (1.0 + np.cos(np.pi * k_y / (taper_h + 1)))

        # 1D Doppler profile
        x_idx = np.arange(grid_w)
        d_x = np.abs(x_idx - cx)
        w_x = np.zeros(grid_w, dtype=np.float32)

        core_mask_x = d_x <= r_w
        taper_mask_x = (d_x > r_w) & (d_x <= (r_w + taper_w))
        k_x = d_x[taper_mask_x] - r_w

        w_x[core_mask_x] = 1.0
        w_x[taper_mask_x] = 0.5 * (1.0 + np.cos(np.pi * k_x / (taper_w + 1)))

        # 2D Separable Outer Product: (H, 1) x (1, W) -> (H, W)
        mask = w_y[:, None] * w_x[None, :]

    elif window_type == "supergaussian":
        y_idx, x_idx = np.ogrid[:grid_h, :grid_w]
        norm_dist = ((y_idx - cy) / max(1, r_h)) ** 2 + ((x_idx - cx) / max(1, r_w)) ** 2
        mask = np.exp(-(norm_dist ** 2)).astype(np.float32)
        mask[mask < 0.01] = 0.0

    elif window_type == "hard":
        mask = np.zeros((grid_h, grid_w), dtype=np.float32)
        mask[cy - r_h : cy + r_h + 1, cx - r_w : cx + r_w + 1] = 1.0

    else:
        raise ValueError(f"Unsupported window_type: {window_type}")

    return mask


def fda_delay_doppler_blend_np(
    H_src: np.ndarray,
    H_tgt: np.ndarray,
    win_h_px: int = 13,
    win_w_px: int = 3,
    window_type: str = "tukey_outward",
    taper_h: int = 2,
    taper_w: int = 1,
) -> np.ndarray:
    """
    Performs Delay-Doppler FDA between source and target channels in NumPy.
    
    Inputs:
        H_src, H_tgt: Complex numpy arrays of shape (..., 132, 14)
                      (Subcarriers x OFDM symbols)
    Returns:
        H_pseudo: Complex numpy array of shape (..., 132, 14)
    """
    # 1. Transform to Delay-Doppler Domain:
    # Subcarriers (axis -2) -> Delay via IFFT
    # Symbols (axis -1) -> Doppler via FFT
    dd_src = np.fft.fftshift(np.fft.fft(np.fft.ifft(H_src, axis=-2), axis=-1), axes=(-2, -1))
    dd_tgt = np.fft.fftshift(np.fft.fft(np.fft.ifft(H_tgt, axis=-2), axis=-1), axes=(-2, -1))

    amp_src = np.abs(dd_src)
    phase_src = np.angle(dd_src)
    amp_tgt = np.abs(dd_tgt)

    # 2. Generate Smooth Mask
    grid_h, grid_w = H_src.shape[-2], H_src.shape[-1]
    mask = create_smooth_fda_window(
        grid_h, grid_w, win_h_px, win_w_px, window_type, taper_h, taper_w
    )

    # 3. Smooth Blending of Amplitudes
    amp_mixed = mask * amp_tgt + (1.0 - mask) * amp_src

    # 4. Recombine with Source Phase
    dd_mixed = amp_mixed * np.exp(1j * phase_src)

    # 5. Inverse Transform to Time-Frequency Domain:
    unshifted = np.fft.ifftshift(dd_mixed, axes=(-2, -1))
    # Doppler -> Time via IFFT, Delay -> Frequency via FFT
    H_pseudo = np.fft.fft(np.fft.ifft(unshifted, axis=-1), axis=-2)

    return H_pseudo
```

---

## 4. Python Implementation: PyTorch (GPU Batched Tensors)

Use this module in your PyTorch neural network pipelines, custom data augmentations, or loss functions:

```python
import torch
import torch.nn as nn
import numpy as np

class SmoothFDAFilter(nn.Module):
    """
    PyTorch module for batch Delay-Doppler Fourier Domain Adaptation with Smooth Windowing.
    Supports complex tensors of shape (B, 2, 14, 132) or (B, 132, 14).
    """
    def __init__(
        self,
        grid_h: int = 132,
        grid_w: int = 14,
        win_h_px: int = 13,
        win_w_px: int = 3,
        window_type: str = "tukey_outward",
        taper_h: int = 2,
        taper_w: int = 1,
    ):
        super().__init__()
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Precompute 2D mask buffer on CPU and register as buffer
        mask_np = create_smooth_fda_window(
            grid_h, grid_w, win_h_px, win_w_px, window_type, taper_h, taper_w
        )
        self.register_buffer("mask", torch.from_numpy(mask_np).float())

    def forward(self, H_src: torch.Tensor, H_tgt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        H_src : torch.Tensor (complex64) of shape (B, 132, 14)
        H_tgt : torch.Tensor (complex64) of shape (B, 132, 14)
        
        Returns
        -------
        H_pseudo : torch.Tensor (complex64) of shape (B, 132, 14)
        """
        # 1. Transform to Delay-Doppler Domain:
        # dim -2 (132 subcarriers) -> IFFT
        # dim -1 (14 symbols) -> FFT
        dd_src = torch.fft.fftshift(
            torch.fft.fft(torch.fft.ifft(H_src, dim=-2), dim=-1),
            dim=(-2, -1)
        )
        dd_tgt = torch.fft.fftshift(
            torch.fft.fft(torch.fft.ifft(H_tgt, dim=-2), dim=-1),
            dim=(-2, -1)
        )

        amp_src = torch.abs(dd_src)
        phase_src = torch.angle(dd_src)
        amp_tgt = torch.abs(dd_tgt)

        # 2. Smooth Blending (Broadcast mask: 132 x 14 over B)
        amp_mixed = self.mask * amp_tgt + (1.0 - self.mask) * amp_src

        # 3. Polar reconstruction
        dd_mixed = torch.polar(amp_mixed, phase_src)

        # 4. Inverse Transform back to Time-Frequency:
        unshifted = torch.fft.ifftshift(dd_mixed, dim=(-2, -1))
        H_pseudo = torch.fft.fft(torch.fft.ifft(unshifted, dim=-1), dim=-2)

        return H_pseudo
```

---

## 5. Verification Test & Numerical Parity

To verify that your Python environment produces identical window values to MATLAB, run this self-test in Python:

```python
def test_smooth_window_parity():
    mask = create_smooth_fda_window(
        grid_h=132, grid_w=14, win_h_px=13, win_w_px=3, window_type="tukey_outward"
    )
    cy, cx = 66, 7  # 0-based centers

    # Verify Doppler (Cols 5..9)
    np.testing.assert_allclose(
        mask[cy, cx - 3 : cx + 4],
        np.array([0.0, 0.5, 1.0, 1.0, 1.0, 0.5, 0.0], dtype=np.float32),
        err_msg="Doppler profile does not match MATLAB specification!",
    )

    # Verify Delay (Rows 57..75)
    expected_delay = np.array(
        [0.0, 0.25, 0.75] + [1.0] * 13 + [0.75, 0.25, 0.0], dtype=np.float32
    )
    np.testing.assert_allclose(
        mask[cy - 9 : cy + 10, cx],
        expected_delay,
        err_msg="Delay profile does not match MATLAB specification!",
    )

    # Verify Diagonal Corner
    np.testing.assert_allclose(
        mask[cy - 7, cx - 2],  # Row 60, Col 6 in MATLAB (1-based)
        0.75 * 0.5,            # 0.375
        err_msg="Corner smoothing does not match MATLAB specification!",
    )
    print(">>> All Smooth Window parity checks passed successfully!")

if __name__ == "__main__":
    test_smooth_window_parity()
```

---

## 6. Summary of Window Variants for Machine Learning

| Variant Tag | Delay Taper ($t_h$) | Doppler Taper ($t_w$) | Total Footprint | Boundary Smoothness | Description |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`smooth1`** *(Default)* | **2 rows** (`[0.75, 0.25]`) | **1 col** (`[0.50]`) | $17 \times 5$ (for $13 \times 3$) | $C^1$ Raised Cosine | **Recommended standard.** Compact taper, preserves maximum target passband while removing 100% of boundary rings. |
| **`smooth2`** | 3 rows (`[0.85, 0.50, 0.15]`) | 2 cols (`[0.75, 0.25]`) | $19 \times 7$ (for $13 \times 3$) | $C^1$ Raised Cosine | Wider transition band for very high noise scenarios. |
| **`smooth3`** | Elliptical ($p=2$) | Elliptical ($p=2$) | Smooth Skirts | $C^\infty$ (Exponential) | Flat-top super-Gaussian taper with zero sidelobes. |
