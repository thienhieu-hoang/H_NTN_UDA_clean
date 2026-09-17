# Experiment Note: A100 → DUR100

## Model & Dataset Paths

- **Source Trained Model Folder:** `single_dataset\A100_2p18e9_600km_70deg_30kHz\<model>`
- **Source Dataset**: `generatedChan\MATLAB\A100_2p18e9_600km_70deg_30kHz`
- **Target Dataset Folder:** `generatedChan\OpenNTN\DUR100nsFix_LoS_2p18G_600km_70deg_r15km_20to30mps`

---

## Dataset Configurations

| Parameter | Source Dataset | Target Dataset |
| :--- | :--- | :--- |
| **Channel Model** | NTN TDL-A (NLOS) | DUR (Dense Urban, LOS) |
| **Data Generator** | MATLAB | OpenNTN |
| **Carrier Frequency ($f_c$)** | 2.18 GHz | 2.18 GHz |
| **Subcarrier Spacing (SCS)** | 30 kHz | 30 kHz |
| **UE Velocity** | [20, 30] m/s | [20, 30] m/s |
| **Satellite Altitude** | 600 km | 600 km |
| **Elevation Angle** | 70° | 70° |

Same pilot position: | **Pilot Configuration (DM-RS)** | **5G NR DM-RS Type 2, Port 1:**<br>• **Frequency Allocation:** Subcarriers port 1 (`[0, 1, 6, 7]`)  per RB (44 pilots/symbol across 11 RBs)<br>• **Time Allocation:** Mapping Type A, Pos 2 (OFDM symbols 2 and 11, 0-indexed)<br>• **Total Pilot REs:** 88 REs per slot |
