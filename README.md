# Financial Fraud Detection in Financial Transactions (Hybrid AI)

### Authors:
**Mehedi Hasan Fahim**  
**Safeyatur Rahman**  

> _A hybrid Artificial Intelligence approach integrating Classical Machine Learning and Quantum Computing for Financial Fraud Detection in transactional systems._

---

## Abstract
Financial institutions handle billions of digital transactions daily — an environment ripe for sophisticated fraud.  
This thesis presents a **Hybrid AI-based Financial Fraud Detection System** that combines **classical ML models** and **quantum computing** concepts to identify fraudulent behavior with high precision and computational efficiency.

The work is divided into five main phases:
1. Dataset selection and preprocessing  
2. Feature engineering and balancing  
3. Classical ML model training and benchmarking  
4. Quantum feature encoding  
5. Hybrid quantum-classical model comparison  

---

## Workflow Summary

### **Phase 1 — Data Preparation**
- Acquired open-source financial datasets (IBM / Kaggle Credit Card Fraud).  
- Cleaned missing values, standardized timestamps, normalized transaction fields.  
- Created advanced behavioral features (frequency, amount ratios, device/IP patterns).  
- Handled class imbalance with **SMOTE**.

### **Phase 2 — Dataset Processing**
- Built efficient streaming pipelines for handling large `.parquet` data (using PyArrow).  
- Scaled all numeric features and exported **ML-ready datasets**:
  - `HI_Small_ready.parquet`
  - `LI_Small_ready.parquet`
  - `HI_Medium_ready.parquet`
  - `LI_Medium_ready.parquet`

### **Phase 3 — Classical Model Benchmarking**
- Trained three baseline ML models:
  - **Logistic Regression**
  - **Random Forest**
  - **XGBoost**
- Evaluated using AUC-ROC, Precision, Recall, F1-score.  
- Identified **XGBoost** as the top-performing classical model.

---

## Results Summary (Phase 3)

| Dataset | Logistic Regression | Random Forest | XGBoost |
|----------|--------------------:|---------------:|---------:|
| **HI_Medium** | 0.8331 | 0.9743 |  **0.9985** |
| **HI_Small** | 0.7925 | 0.9814 |  **0.9996** |
| **LI_Medium** | 0.7709 | 0.9769 |  **0.9984** |
| **LI_Small** | 0.7007 | 0.9762 |  **0.9994** |

> XGBoost achieved near-perfect AUC across all datasets, establishing a strong classical baseline.

---

##  Folder Structure


FFD_Thesis/
│
├── data/
│ ├── raw/ ← Original data (not committed)
│ ├── interim/ ← Temporary files
│ ├── processed/ ← Cleaned parquet files
│ ├── ready/ ← Final ML-ready datasets
│
├── notebooks/
│ ├── 01_load_and_explore.ipynb
│ ├── 02_preprocessing_all.ipynb
│ ├── 03_classical_baseline.ipynb
│ └── 04_quantum_phase.ipynb
│
├── models/ ← Trained models & scalers
├── reports/ ← Evaluation metrics, plots
├── src/ ← Python source scripts
├── requirements.txt
├── prep_medium_streaming.py
├── README.md
└── .gitignore


##  Running Locally (Step-by-Step)

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/mhfahim/FFD_Thesis.git
cd FFD_Thesis
```

### 2️⃣ Create a Virtual Environment

```bash
python -m venv .venv
```

### 3️⃣ Activate the Virtual Environment

#### On Windows

```powershell
.\.venv\Scripts\activate
```

#### On macOS / Linux

```bash
source .venv/bin/activate
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 5️⃣ Launch Jupyter Lab

```bash
jupyter lab
```

### Then open

notebooks/02_preprocessing_all.ipynb → for data preparation

notebooks/03_classical_baseline.ipynb → for classical model evaluation

### Reproduction of Experiments

#### Data Preparation

```bash
python prep_medium_streaming.py
```

#### Model Training

Open the notebook:

```
notebooks/03_classical_baseline.ipynb
```

