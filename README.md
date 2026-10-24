# ⚡ EcoGrid AI — Predictive Circular-Resource Grid (NexoraGrid)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Enabled-orange.svg)](https://xgboost.ai/)
[![Tests Passing](https://img.shields.io/badge/Tests-11%2F11%20Passed-brightgreen.svg)]()

> **Predictive Circular-Resource Grid for a Multi-Building Campus**  
> An end-to-end AI platform monitoring, forecasting, detecting anomalies, attributing root causes, and closing circular loops for electricity, water, solar generation, and waste across campus facilities.

GitHub Repository: [https://github.com/Rohith3014/NexoraGrid](https://github.com/Rohith3014/NexoraGrid)

---

## 📌 Executive Summary & Key Achievements

| Metric / Constraint | Target Requirement | Measured Achievement | Status |
|---|---|---|:---:|
| **Edge Waste Classifier Accuracy** | > 85.0% across ≥ 5 classes | **85.35% Validation Accuracy** (6 classes) | ✅ **EXCEEDED** |
| **Edge Inference Latency** | < 2,000 ms per event | **0.021 ms per item** (~21 microseconds) | ✅ **100,000x FASTER** |
| **Macro F1-Score** | Balanced performance | **0.8360 Macro F1** | ✅ **VERIFIED** |
| **Short-Horizon Forecasting** | 6-hour & 12-hour ahead | **24 XGBoost models** trained across 8 facilities | ✅ **COMPLETE** |
| **Forecasting Accuracy vs Baseline** | Lower than naive lag-1 | **45% to 79.5% Error Reduction** (MAE / RMSE) | ✅ **SUPERIOR** |
| **Zero-Waste Diversion Rate** | Sustainable campus target | **72.24% Diversion Rate** (90,309 kg diverted) | ✅ **VERIFIED** |
| **Organic Composting Loop** | Strict physical mass ratio | **17,474.55 kg Compost** from 69,898 kg wet waste (4:1) | ✅ **CONSERVED** |
| **Intermittent Connectivity & Buffering**| Zero dashboard corruption | **Local SQLite WAL queue with deduplication** | ✅ **TESTED** |
| **Per-Building Hardware Budget** | Under team budget stated | **₹11,500 (~$140)** per building (budget < ₹15,000) | ✅ **FEASIBLE** |
| **Automated Test Coverage** | 100% core pass | **11 / 11 Pytest Tests Passed (100%)** | ✅ **VERIFIED** |

---

## 🏗️ System Architecture

```mermaid
graph TD
    subgraph Edge Layer [Building Edge Nodes (B01 - B08)]
        S1[Precision Load Cell] --> EB[Edge Micro-Controller]
        S2[Dielectric Moisture Probe] --> EB
        S3[Inductive Metal Proximity] --> EB
        S4[Optical Transparency Sensor] --> EB
        S5[Piezo Vibration Impact Sensor] --> EB
        CT[Smart CT Energy Clamp & Water Flow] --> EB
        TS[Turnstile Footfall Sensor] --> EB
        
        EB --> WC[XGBoost Edge Waste Classifier < 0.02ms]
        EB --> SQ[Local SQLite Buffer Queue]
    end

    subgraph Transport Layer [Fault-Tolerant Communications]
        SQ -->|MQTT / HTTP Sync| GW{Network Connectivity}
        GW -->|Online| DR[Idempotent Deduplication Engine]
        GW -->|Dropout| LB[Local Flash Storage Buffer]
        LB -->|Reconnect Burst| DR
    end

    subgraph Central Intelligence Grid
        DR --> DB[(Campus SQLite / TimeSeries DB)]
        DB --> FC[6h & 12h Recursive Forecasters]
        DB --> AD[MAD & Multivariate Anomaly Detector]
        DB --> RC[Multi-Source Root Cause Analyzer]
        DB --> CA[Circular Economy Mass & Cost Accounting]
        DB --> IN[Building Sustainability Scoring]
    end

    subgraph Presentation Layer
        FC & AD & RC & CA & IN --> UI[Streamlit Interactive Dashboard]
    end
```

---

## 🚀 Quickstart & Local Execution

### 1. Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- `pip` package manager

### 2. Clone the Repository
```bash
git clone https://github.com/Rohith3014/NexoraGrid.git
cd NexoraGrid
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Full Data & Training Pipeline
Executes dataset normalization, trains the edge waste classifier, fits 24 multi-building forecasters, detects anomalies, calculates circular accounting, and populates SQLite:
```bash
python scripts/train_and_process.py
```

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`** to interact with the live dashboard!

### 6. Run the Automated Test Suite
```bash
python -m pytest -v tests/test_ecogrid.py
```

---

## 📊 Core Features & Operational Modules

### 1. ♻️ Smart-Bin Multi-Modal Edge Waste Classifier
- **Sensory Fusion:** Fuses 5 asynchronous physical sensors: load cell mass (`weight_g`), surface & core moisture (`moisture_pct`), inductive proximity (`metal_signal`), optical transmittance (`transparency`), and impact resonance (`acoustic_hardness`).
- **6 Classes Classified:** `food_organic`, `paper_cardboard`, `plastic`, `glass`, `metal`, `ewaste`.
- **Latency & Compute:** Runs inference in **0.021 ms**, enabling immediate mechanical flap actuation at the receptacle edge.

### 2. 📈 6-Hour & 12-Hour Recursive Resource Forecasting
- **Short-Horizon Spike Detection:** Predicts surges prior to student meal rushes (Food Court B07) or exam week study blocks (Library B03 & Hostels B05/B06).
- **Edge Accuracy-Compute Tradeoff:** Leverages Gradient Boosted Trees with cyclical temporal encodings (sin/cos of hour and day) and rolling lag statistics ($t-1, t-2, t-6, t-12, t-24$). Memory footprint is $<9\text{ MB}$, eliminating the need for power-hungry cloud GPU instances.

### 3. 🚨 Anomaly Detection & Contextual Root-Cause Attribution
- Detects rolling statistical deviations and abnormal overnight baselines (e.g. nocturnal water leaks or HVAC equipment left running during semester breaks).
- **Multi-Source Evidence Attribution:** Combines telemetry with academic calendar events, ambient weather feeds, and manual facilities maintenance logs to present a ranked probable cause and a concrete mitigation recommendation.

### 4. 🔄 Circular Economy Accounting & Zero-Waste Ledger
- Tracks 120 days of campus operations across 8 facilities.
- **Mass Flow Accounting:**
  - Total waste: **125,006.1 kg**
  - Diverted from landfill: **90,309.0 kg (72.24%)**
  - Composted: **69,898.2 kg** $\to$ **17,474.55 kg** finished organic compost (strictly adhering to the 4:1 mass conservation ratio).
  - Recycled: **20,410.8 kg**
  - Residual Landfill: **35,339.4 kg**
- **Economic & Carbon Ledger:** Transparent calculation separating direct utility tariffs (₹8.00/kWh) and physical measurements from standardized grid emission estimates ($0.70\text{ kg CO}_2/\text{kWh}$).

### 5. 🏆 Building Sustainability Scoring & Behavioral Incentives
- Transparent, weighted scorecard:
  - Electricity efficiency: 25%
  - Water efficiency: 25%
  - Waste diversion rate: 25%
  - Anomaly resolution: 15%
  - Data reporting fidelity: 10%
- Closed-loop incentives: Hostel Inter-Block Eco-Cup (reallocating utility budget savings) and dynamic plate-waste signage in cafeterias.

---

## 🛠️ Hardware Bill of Materials (BOM) — < ₹15,000 / $180 Budget

| Component | Hardware Specification | Purpose | Unit Cost (INR) |
|---|---|---|:---:|
| **Edge Compute Node** | Raspberry Pi 4 (2GB) / Orange Pi 3 | Local ML inference & buffer | ₹4,200 |
| **Precision Load Cell** | 4-Wire Strain Gauge + HX711 24-bit ADC | Disposed mass measurement | ₹650 |
| **Moisture Probe** | Capacitive Analog Dielectric Probe | Wet vs dry waste distinction | ₹350 |
| **Inductive Proximity Sensor** | NPN 6–36V Inductive Probe | Electromagnetic metal detection | ₹450 |
| **Optical Transmittance Pair** | Photodiode & IR Emitter Pair | Transparency coefficient | ₹250 |
| **Piezo Acoustic Sensor** | Piezoelectric Disc Element | Material hardness & resonance | ₹300 |
| **Submetering CT Clamps** | Non-Invasive Current Transformer + Flow Meter | Electrical load & water flow | ₹3,800 |
| **Weatherproof Enclosure & Wiring** | IP65 ABS Enclosure + Power Supply | Physical deployment | ₹1,500 |
| **Software Stack** | Linux + Python + SQLite + XGBoost | Completely Open-Source | ₹0 |
| **Total Hardware Cost per Building** | **Complete Edge Node BOM** | | **₹11,500 (~$140)** |

---

## 📈 Horizontal Scaling Argument (8 to 30+ Buildings)

1. **Edge Decentralization:** Feature extraction and inference occur locally on the edge node. Central telemetry transfers require only **~86.4 KB/day** across the entire 30-building campus.
2. **Storage Architecture:** SQLite WAL mode handles tens of thousands of continuous writes with sub-millisecond latencies. For campus expansions beyond 100 buildings, the schema trivially ports to PostgreSQL / TimescaleDB.

---

## 🛡️ Critical Failure Modes & Concrete Mitigations

1. **Failure Mode 1: Sensor Drift & Calibration Variance**
   - *Risk:* Environmental humidity and residue buildup degrade dielectric moisture probes and zero-load strain gauges over a semester.
   - *Mitigation:* Daily automated zero-point tare calibration during 03:00–05:00 empty-bin windows; automated drift detection alerts facilities when baseline variance exceeds 15%.
2. **Failure Mode 2: Occupant Gaming of the Incentive System**
   - *Risk:* Students disposing heavy non-recyclable inert matter (e.g. rocks, soaked paper) to inflate diversion mass rankings.
   - *Mitigation:* Cross-sensor sanity verification flags anomalies when weight is high ($>1\text{ kg}$) but transparency and dielectric moisture contradict known organic or recyclable signatures.

---

## 👨‍💻 Author & Project Info
- **Project:** EcoGrid AI / NexoraGrid
- **Repository:** [https://github.com/Rohith3014/NexoraGrid](https://github.com/Rohith3014/NexoraGrid)
- **Built for:** Multi-Building Campus Predictive Circular-Resource Grid
