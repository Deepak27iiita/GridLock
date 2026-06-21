<div align="center">
  <h1>🚦 GridLock</h1>
  <h3>AI-Driven Parking Intelligence & Congestion Impact Platform</h3>
  <p>Advanced parking hotspot detection and congestion impact scoring for Bengaluru traffic enforcement.</p>
</div>

---

## 📖 Overview

**GridLock** is an end-to-end AI parking intelligence platform built for urban traffic enforcement. By ingesting and analyzing historical police violation records, GridLock automatically detects illegal parking hotspots, quantifies their impact on traffic flow using a **Parking Congestion Impact Score (PCIS)**, forecasts temporal violation patterns, and generates optimized patrol deployment plans.

Instead of reactive enforcement, GridLock enables proactive congestion management by answering:
- **WHERE** does parking hurt traffic the most?
- **WHEN** will the violations peak?
- **WHO** should deploy where for maximum delay reduction?

---

## ✨ Key Features

- 📍 **Geospatial Hotspot Detection**: Utilizes Uber's H3 geospatial indexing (Resolution 9) combined with clustering algorithms to map critical zones.
- 📉 **Parking Congestion Impact Score (PCIS)**: An ML-enhanced (LightGBM + Isotonic Calibration) score from 0.0 to 1.0 ranking the severity of congestion caused by parking hotspots.
- 🕒 **Temporal Forecasting**: Predicts violation counts by hour using an ensemble of LightGBM and historical lookup tables.
- 👮 **Enforcement Recommender**: Generates time-windowed patrol deployments and tow-truck assignments based on available officers and PCIS rankings.
- 🕹️ **What-If Simulator**: Allows command center operators to simulate the impact of deploying extra enforcement teams and estimate the delay reduction.
- 📊 **Command Dashboard**: A responsive, dark-themed React-style frontend featuring live PCIS heatmaps, KPIs, charts, and interactive forecasting.

---

## 🏗️ System Architecture & Tech Stack

### Architecture
GridLock follows a robust 3-tier architecture:
1. **Data Layer**: Processes anonymized police violation records.
2. **ML Engine**: Powers the Hotspot Clustering, PCIS calculation, and Temporal Forecasting.
3. **Application Layer**: Exposes predictions via a FastAPI backend and visualizes them on an interactive HTML/JS frontend.

### Technology Stack
- **Backend**: Python 3.12, FastAPI, Uvicorn
- **Machine Learning**: LightGBM, scikit-learn, Pandas, NumPy
- **Geospatial & Storage**: H3 (Uber), PyArrow (Parquet), Joblib
- **Frontend**: HTML5/CSS3, Vanilla JavaScript, Leaflet (Maps), Chart.js

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.11 or 3.12
- `pip` package manager
- Recommended: 4GB+ RAM

### Installation & Execution

1. **Clone the repository and install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the ML Training Pipeline** (Takes ~45-65 seconds):
   ```bash
   python scripts/train_pipeline.py
   ```
   *Note: This processes the raw data, builds the H3 cells, trains the models, and generates evaluation metrics.*

3. **(Optional) Rescore PCIS**:
   ```bash
   python scripts/rescore_pcis.py
   ```
   *Fast utility to rescore and apply percentile spreading for sharper risk levels.*

4. **Start the FastAPI Backend**:
   ```bash
   python backend/run.py
   ```

5. **Open the Dashboard**:
   Navigate to **[http://localhost:8000](http://localhost:8000)** in your web browser.

---

## 🗺️ Hackathon Demo Flow

To effectively demonstrate GridLock's capabilities, follow this flow:
1. **Show PCIS Heatmap**: Point out the red zones which indicate the highest congestion impact (Severe/Critical).
2. **Open Priority Hotspots**: Review the top detected junction and corridor clusters.
3. **Click "Generate Plan"**: Demonstrate the time-windowed patrol recommendations.
4. **Run What-If Simulator**: Show how adding extra teams quantifies delay reductions.
5. **Show Model Accuracy**: Highlight metrics such as Silhouette score, MAPE, and R² on hold-out data.

---

## 📂 Project Structure

```text
GridLock/
├── Dataset/                   # Raw anonymized violation records
├── backend/                   # FastAPI backend & ML services
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   └── services/          # ETL, Clustering, PCIS, Recommender
│   └── run.py                 # Backend server entry point
├── data/                      # Processed data & trained ML models
├── frontend/                  # Dashboard (HTML, JS, CSS)
├── scripts/                   # Training and rescoring pipelines
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🔮 Future Enhancements
- Live ANPR camera integration for real-time parking detection.
- YOLOv8 stationary vehicle detection in no-parking zones.
- Mobile patrol app for field officers.

---
*Developed for Bengaluru Traffic Enforcement — Hackathon 2025*