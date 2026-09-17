# Experiment Note: A100 (Port 1) → DUR100 (Port 2)

## Model & Dataset Paths

- **Source Trained Model Folder:** `single_dataset\A100_2p18e9_600km_70deg_30kHz\<model>`
- **Source Dataset**: `generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz`
- **Target Dataset Folder:** `generatedChan\OpenNTN\DUR100nsFix_LoS_port2_Apos2_2p18G_600km_70deg_r15km_20to30mps`

---

## Dataset Configurations

| Parameter | Source Dataset | Target Dataset |
| :--- | :--- | :--- |
| **Channel Model** | NTN TDL-A (NLOS) | DUR (Dense Urban, LoS) |
| **Data Generator** | MATLAB | OpenNTN |
| **Carrier Frequency ($f_c$)** | 2.18 GHz | 2.18 GHz |
| **Subcarrier Spacing (SCS)** | 30 kHz | 30 kHz |
| **UE Velocity** | [20, 30] m/s | [20, 30] m/s |
| **Satellite Altitude** | 600 km | 600 km |
| **Elevation Angle** | 70° | 70° |
| **Pilot Pattern (DM-RS)** | **5G NR DM-RS Type 2, Port 1:**<br>• **Frequency:** Port 1 subcarriers `[0, 1, 6, 7]` per RB<br>• **Time:** Mapping Type A, Pos 2 (Symbols 2 & 11, 0-indexed)<br>• **Total Pilots:** 88 REs per slot | **5G NR DM-RS Type 2, Port 2:**<br>• **Frequency:** Port 2 subcarriers `[2, 3, 8, 9]` per RB<br>• **Time:** Mapping Type A, Pos 2 (Symbols 2 & 11, 0-indexed)<br>• **Total Pilots:** 88 REs per slot |

> [!NOTE]
> **Pilot Port Shift Experiment:**
> This experiment evaluates model transferability across antenna ports (subcarrier shift with identical symbol timing):
> - **Frequency Domain:** Port 1 (`[0, 1, 6, 7]`) $\rightarrow$ Port 2 (`[2, 3, 8, 9]`)
> - **Time Domain:** Mapping Type A Pos 2 (Symbols 2 & 11, 0-indexed) on both domains

