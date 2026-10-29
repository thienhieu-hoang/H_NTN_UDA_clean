In OFDM wireless channels, individual complex channel entries $H(k, l)$ will almost always follow a standard **Rayleigh/Rician distribution** in magnitude and a **uniform distribution $[-\pi, \pi]$** in phase across Source, Target, and Pseudo datasets. Consequently, magnitude/phase histograms look nearly identical and **completely hide the domain gap** (delay spread, Doppler shift, and frequency selectivity).

To demonstrate and publish that the physical mathematical transformation (Approach A) successfully bridges the domain gap, you should report **Delay-Doppler (DD) domain profiles and autocorrelation metrics**.

---

### What You Should Report

```
                             3 CORE DOMAIN CHARACTERIZATIONS
 ┌───────────────────────────┬───────────────────────────┬───────────────────────────┐
 │ 1. Delay Domain (PDP)     │ 2. Doppler Domain         │ 3. Autocorrelation (ACF)  │
 ├───────────────────────────┼───────────────────────────┼───────────────────────────┤
 │ • Power Delay Profile     │ • Doppler Power Spectrum  │ • Frequency ACF:          │
 │   P(τ) = E[|H(τ, ν)|²]    │   S(ν) = E[|H(τ, ν)|²]    │   Rf(Δk) → Coherence BW   │
 │ • Shows τ_rms expansion   │ • Shows Doppler shift Δν  │ • Time ACF:               │
 │   (100 ns → 300 ns)       │   and spread              │   Rt(Δl) → Coherence Time │
 └───────────────────────────┴───────────────────────────┴───────────────────────────┘
```

---

### Detailed Reporting Breakdown

#### 1. Delay Domain: Power Delay Profile (PDP) $P(\tau)$
* **How to compute**: Take IFFT across subcarriers ($f \to \tau$):
  $$h(\tau, t) = \text{IFFT}_{f}[H(f, t)], \quad P(\tau) = \frac{1}{N_{\text{sym}}} \sum_{t} \mathbb{E}\left[|h(\tau, t)|^2\right]$$
* **What it proves**: 
  - **Source (`A100`)**: Narrow delay profile concentrated in $[0, 100\text{ ns}]$.
  - **Target (`DUR300`)**: Wide delay profile extending out to $300\text{ ns}$.
  - **Pseudo**: Proves that the $\beta_\tau \approx 3$ dilation accurately stretched the delay dispersion to match the target's multipath decay tail.

#### 2. Doppler Domain: Doppler Power Spectrum $S(\nu)$
* **How to compute**: Take FFT across OFDM symbols ($t \to \nu$):
  $$\mathcal{H}(\tau, \nu) = \text{FFT}_{t}[h(\tau, t)], \quad S(\nu) = \frac{1}{N_{\text{sc}}} \sum_{\tau} \mathbb{E}\left[|\mathcal{H}(\tau, \nu)|^2\right]$$
* **What it proves**:
  - Shows that the bulk Doppler shift $\Delta\nu$ (caused by satellite orbital motion and UE speed) was shifted to align with the Target domain center frequency, rather than remaining at the Source's center.

#### 3. Correlation Structure: Frequency & Time Autocorrelation
* **Frequency Autocorrelation $R_f(\Delta k)$**:
  $$R_f(\Delta k) = \mathbb{E}\left[ H(k + \Delta k, l) \, H^*(k, l) \right]$$
  - Directly demonstrates the **Coherence Bandwidth ($B_c$)**.
  - In Source ($100\text{ ns}$), $R_f(\Delta k)$ decays slowly ($B_c$ is large).
  - In Target ($300\text{ ns}$) and Pseudo, $R_f(\Delta k)$ decays **$3\times$ faster** ($B_c$ is compressed). Plotting curves of Source vs. Pseudo vs. Target side-by-side provides proof of physical fidelity.
* **Time Autocorrelation $R_t(\Delta l)$**:
  $$R_t(\Delta l) = \mathbb{E}\left[ H(k, l + \Delta l) \, H^*(k, l) \right]$$
  - Directly demonstrates the **Coherence Time ($T_c$)** and Doppler phase velocity.

---

### Recommended Summary Table for Your Report

Include this comparative table in your report/note:

| Physical Metric | Source (`A100`) | Target (`DUR300`) | Math Pseudo (`A100_DUR300_LS`) |
| :--- | :---: | :---: | :---: |
| **RMS Delay Spread ($\tau_{\text{rms}}$)** | $\sim 100\text{ ns}$ | $\sim 300\text{ ns}$ | $\sim 300\text{ ns}$ *(matches Target)* |
| **Coherence Bandwidth ($B_{c, 0.5}$)** | Wide ($\sim 2\text{ MHz}$) | Compressed ($\sim 0.67\text{ MHz}$) | Compressed *(matches Target)* |
| **Average Doppler Shift ($\bar{\nu}$)** | Source center | Target center ($\Delta\nu$ offset) | Target center *(matches Target)* |
| **2D Linear Interp. NMSE (dB)** | (Source baseline) | (Target baseline) | Evaluated baseline |

---

### Recommended Visualization Figure (For Paper / Worklog)
A standard **4-panel figure** comparing:
1. **Panel (a)**: 1D Power Delay Profile $P(\tau)$ (Source vs. Pseudo vs. Target).
2. **Panel (b)**: 1D Doppler Spectrum $S(\nu)$ (Source vs. Pseudo vs. Target).
3. **Panel (c)**: Frequency Autocorrelation $|R_f(\Delta k)|$ (demonstrating $B_c$ alignment).
4. **Panel (d)**: Qualitative 2D Channel Heatmaps of a sample from each domain.