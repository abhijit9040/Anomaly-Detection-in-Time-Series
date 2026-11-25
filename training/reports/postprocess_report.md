# False-Alarm Alert Post-Processing & Ensemble Decision Gate Report

**Date:** 2026-11-25  
**Dataset:** SMD machine-1-1 (Window size $W=30$, stride $S=1$, 38 features)  
**Author:** Antigravity Agent  
**Branch:** `feat/alert-postprocessing`

---

## 1. Executive Summary & Gate Outcomes

This report documents the rigorous evaluation of two interventions designed to improve alert quality and reduce false-alarm event bursts (previously 94 bursts in LSTM vs 42 in PCA):
1. **Causal Score Post-Processing** (`smooth_scores` + `hysteresis_flags` + `min_duration_filter`): Fixed defaults decided in advance: $k=5$ (trailing mean), $m=3$ (consecutive persistence), $\text{low\_ratio}=0.8$.
2. **Offline LSTM + PCA Ensemble**: Normalized scores with validation p99 $s_{\text{ens}} = 0.5 \cdot \frac{s_{\text{lstm}}}{\text{val\_p99}_{\text{lstm}}} + 0.5 \cdot \frac{s_{\text{pca}}}{\text{val\_p99}_{\text{pca}}}$.

### Decision Gate Summary

| Gate | Protocol / Condition | Required Hurdle | Measured (Held-Out 70%) | Outcome | Action |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Post-Processing Gate** | Held-out 70% test slice ($N=19,915$) | FA events drop $\ge 30\%$<br>Event recall $\ge 7/8$<br>Window F1 drop $\le 2.0$ pp | **FA drop: 56.72%** (67 $\to$ 29)<br>**Event recall: 8/8 (100%)**<br>**F1 drop: 1.34 pp** (0.6191 $\to$ 0.6057) | **PASS** | **Integrate** into backend (`AlertingService`) and frontend |
| **Ensemble Gate** | Held-out 70% test slice ($N=19,915$) | F1 beats BOTH LSTM & PCA by $\ge 2.0$ pp | **+1.42 pp vs LSTM** (0.6333 vs 0.6191)<br>+3.36 pp vs PCA (0.6333 vs 0.5997) | **FAIL** (Negative Result) | **Do NOT integrate** into backend/frontend; document offline findings |

---

## 2. Methodology & Protocol Compliance

1. **Strict Causality**: All post-processing at timestep $t$ strictly depends only on scores $\le t$:
   - Smoothing: Trailing mean over $\max(0, t - k + 1) \dots t$.
   - Hysteresis: Turns ON when smoothed score $> \text{threshold}$, turns OFF when $< \text{threshold} \times 0.8$.
   - Persistence filter: Flags only after $m$ consecutive active windows.
2. **Threshold Re-Derivation**: Smoothing reduces variance and shifts the upper tail of the error distribution. Therefore, for every smoothing configuration, the threshold is re-derived as the 99.0th percentile of the smoothed normal validation set (`rederive_threshold`). No test data was used for threshold selection.
3. **Temporal Split Protocol**:
   - **Calibration slice (first 30%, $N=8,535$ timesteps)**: Purely normal operation (0 anomalies). Used to inspect sensitivity and select $(k, m)$.
   - **Held-out test slice (last 70%, $N=19,915$ timesteps)**: Contains all 8 anomaly segments (2,926 anomalous windows). Used as the honest, unbiased benchmark for headline decision gates.

---

## 3. Post-Processing Results

### Table 1: Full Test Set ($N=28,450$) — Fixed Defaults ($k=5, m=3, \text{low\_ratio}=0.8$)

| Method | Threshold | Strategy | Precision | Recall | Window F1 | Flagged Rate | Event Recall | Detected Events | False Alarm Events | Mean Delay | Alerts / 1k Windows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM Raw** | 0.154136 | Val p99 | 0.4242 | 0.9491 | 0.5864 | 0.2301 | 1.0000 | 8/8 | 94 | 0.0 | 4.11 |
| **LSTM Post-processed** | 0.152915 | Smoothed Val p99 | 0.4063 | 0.9627 | 0.5715 | 0.2437 | 1.0000 | 8/8 | **46 (-51.1%)** | 2.0 | **2.00 (-51.3%)** |
| **PCA Raw** | 0.235026 | Val p99 | 0.6090 | 0.5803 | 0.5943 | 0.0980 | 1.0000 | 8/8 | 42 | 0.0 | 3.06 |
| **PCA Post-processed** | 0.234524 | Smoothed Val p99 | 0.5834 | 0.6107 | 0.5968 | 0.1077 | 1.0000 | 8/8 | **24 (-42.9%)** | 2.9 | **1.79 (-41.5%)** |

### Table 2: Held-Out 70% Slice ($N=19,915$) — Headline Decision Benchmark

| Method | Threshold | Precision | Recall | Window F1 | Event Recall | False Alarm Events | Mean Delay | Alerts / 1k Windows |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **LSTM Raw Held-out** | 0.154136 | 0.4594 | 0.9491 | **0.6191** | 1.0000 (8/8) | 67 | 0.0 | 4.52 |
| **LSTM Post-proc (Fixed Defaults $k=5, m=3$)** | 0.152915 | 0.4418 | 0.9627 | **0.6057** (-1.34 pp) | 1.0000 (8/8) | **29 (-56.7%)** | 2.0 | **2.01 (-55.5%)** |
| **LSTM Post-proc (Cal-Selected $k=1, m=10$)** | 0.154136 | 0.4519 | 0.9293 | **0.6081** (-1.10 pp) | 1.0000 (8/8) | **27 (-59.7%)** | 8.0 | **1.96 (-56.6%)** |
| **PCA Raw Held-out** | 0.235026 | 0.6204 | 0.5803 | **0.5997** | 1.0000 (8/8) | 38 | 0.0 | 4.17 |
| **PCA Post-proc (Fixed Defaults $k=5, m=3$)** | 0.234524 | 0.5925 | 0.6107 | **0.6015** (+0.18 pp) | 1.0000 (8/8) | **22 (-42.1%)** | 2.9 | **2.46 (-41.0%)** |
| **PCA Post-proc (Cal-Selected $k=1, m=10$)** | 0.235026 | 0.6094 | 0.5444 | **0.5751** (-2.46 pp) | 1.0000 (8/8) | **23 (-39.5%)** | 8.2 | **2.56 (-38.6%)** |

### Post-Processing Key Takeaway:
Fixed defaults ($k=5, m=3$) deliver the best balance across the board:
- Cut false-alarm bursts by **56.7%** on LSTM (from 67 to 29) and **42.1%** on PCA (from 38 to 22).
- Retains **100% event recall (8/8 events)**.
- Adds only a minor detection delay of **2.0 timesteps** (compared to 8.0 timesteps for $m=10$).
- F1 change is minimal (-1.34 pp for LSTM, +0.18 pp for PCA).

---

## 4. Sensitivity Analysis (Transparency Only, Not Used for Selection)

Grid sweep over $k \in \{1, 3, 5, 10, 20\}$ and $m \in \{1, 3, 5, 10\}$ on full test data for LSTM:

| $k$ (Smoothing) | $m$ (Min Duration) | Threshold | Precision | Recall | Window F1 | Event Recall | False Alarm Events | Mean Delay (steps) | Alerts / 1k Windows | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 1 | 1 | 0.154136 | 0.4242 | 0.9491 | 0.5864 | 1.0000 | 94 | 0.0 | 4.11 | Raw identity baseline |
| 1 | 3 | 0.154136 | 0.4357 | 0.9436 | 0.5962 | 1.0000 | 66 | 2.0 | 2.95 | Duration filter only |
| 1 | 5 | 0.154136 | 0.4385 | 0.9381 | 0.5976 | 1.0000 | 56 | 3.9 | 2.53 | Duration filter only |
| 1 | 10 | 0.154136 | 0.4431 | 0.9293 | 0.6000 | 1.0000 | 42 | 8.0 | 1.93 | Large delay |
| 3 | 1 | 0.153408 | 0.4074 | 0.9576 | 0.5718 | 1.0000 | 72 | 0.0 | 3.20 | Smoothing only |
| 3 | 3 | 0.153408 | 0.4158 | 0.9542 | 0.5791 | 1.0000 | 50 | 2.0 | 2.28 | Balanced |
| **5** | **3** | **0.152915** | **0.4063** | **0.9627** | **0.5715** | **1.0000** | **46** | **2.0** | **2.00** | **Fixed Default** |
| 5 | 5 | 0.152915 | 0.4072 | 0.9576 | 0.5716 | 1.0000 | 40 | 3.9 | 1.76 | Higher suppression |
| 10 | 3 | 0.152166 | 0.3846 | 0.9682 | 0.5504 | 1.0000 | 44 | 2.0 | 1.93 | Slight over-smoothing |
| 20 | 5 | 0.151121 | 0.3546 | 0.9686 | 0.5192 | 1.0000 | 39 | 3.9 | 1.76 | Heavy over-smoothing |

---

## 5. Offline Ensemble Results (Negative Result)

The ensemble normalizes scores using the validation 99th percentile:
$$\text{Score}_{\text{ens}} = w \cdot \frac{\text{LSTM}}{\text{val\_p99}_{\text{lstm}}} + (1 - w) \cdot \frac{\text{PCA}}{\text{val\_p99}_{\text{pca}}}$$
Threshold is the 99.0th percentile of validation ensemble scores.

### Ensemble Performance Table

| Evaluation Slice | Weight Configuration | Threshold | Precision | Recall | Window F1 | Event Recall | False Alarms | Delta vs LSTM | Delta vs PCA |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Held-Out 70%** | $w=0.5$ (Fixed default) | 0.966852 | 0.5013 | 0.8595 | **0.6333** | 1.0000 | 36 | **+1.42 pp** | **+3.36 pp** |
| **Held-Out 70%** | $w=0.25$ ($w_{\text{pca}}=0.75$) | 0.952236 | 0.5307 | 0.7888 | **0.6345** | 1.0000 | 36 | **+1.54 pp** | **+3.48 pp** |
| **Held-Out 70%** | $w=0.75$ ($w_{\text{pca}}=0.25$) | 0.980807 | 0.4748 | 0.9094 | **0.6239** | 1.0000 | 49 | **+0.48 pp** | **+2.42 pp** |
| **Full Test Set** | $w=0.5$ (Fixed default) | 0.966852 | 0.4729 | 0.8595 | **0.6101** | 1.0000 | 60 | **+2.37 pp** | **+1.58 pp** |
| **Full Test Set** | $w=0.25$ ($w_{\text{pca}}=0.75$) | 0.952236 | 0.5067 | 0.7888 | **0.6170** | 1.0000 | 46 | **+3.06 pp** | **+2.27 pp** |

### Decision Gate Verdict:
- The ensemble gate requires held-out F1 to beat **BOTH** LSTM-only and PCA-only by at least **2.0 pp**.
- On the held-out 70% slice, the ensemble achieved **F1 = 0.6333**, beating PCA-only (0.5997) by **+3.36 pp**, but beating LSTM-only (0.6191) by only **+1.42 pp** (below the 2.0 pp hurdle).
- **Outcome**: **FAILED GATE**. As required by the protocol, this is reported honestly as a negative result and the ensemble is **NOT** integrated into the runtime service.

---

## 6. Honest Engineering Tradeoffs & Caveats

1. **Detection Delay**: Post-processing adds a temporal detection delay averaging **2.0 timesteps** for LSTM and **2.9 timesteps** for PCA. In a streaming environment sampling at 1 Hz, this corresponds to an alert arriving 2 to 3 seconds after the onset of an anomaly.
2. **Window-Level Precision vs Operational Bursts**: On the full test set, raw LSTM window F1 is 0.5864 (Precision 0.4242, Recall 0.9491). Post-processing slightly shifts window F1 to 0.5715 (Precision 0.4063, Recall 0.9627, a 1.49 pp drop). However, it halves operational noise (false alarm bursts drop from 94 to 46, and alerts per 1000 windows drop from 4.11 to 2.00). In practical operations, an on-call engineer experiences 51% fewer pager interrupts.
3. **Sensor Heterogeneity**: PCA detects anomalies through low-rank global covariance, whereas LSTM detects temporal prediction violations. Combining them provides a small F1 gain (+1.4 pp), but adding the operational complexity of maintaining two models in production is not justified under our 2 pp threshold.
