# 🚀 VentureVision AI
## Startup Outcome Prediction & Decision Support System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-green)

---

## 📋 Overview

VentureVision AI is a production-ready machine learning system that predicts startup outcomes — **IPO**, **Acquisition**, or **Failure** — based on key business metrics. It combines ensemble ML models with an interactive Streamlit dashboard featuring AI-driven insights, what-if simulations, and a rich analytics suite.

---

## 🗂️ Project Structure

```
venturevision/
├── app.py                     # Streamlit entry point
├── train_model.py             # Full training pipeline (run this first)
├── requirements.txt
├── README.md
│
├── data/
│   └── startup_success_dataset.csv   # Kaggle dataset (100K rows)
│
├── models/                    # Auto-created by train_model.py
│   ├── model.pkl              # Best trained model (Random Forest)
│   ├── scaler.pkl             # StandardScaler
│   ├── encoders.pkl           # LabelEncoders for categorical features
│   ├── label_encoder.pkl      # Target LabelEncoder
│   ├── feature_names.pkl      # Feature name list
│   └── model_results.pkl      # All model metrics + test predictions
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py       # Data loading, cleaning, encoding, scaling
│   ├── eda.py                 # All Plotly visualisation helpers
│   └── model_utils.py         # Model I/O, prediction, insights, charts
│
└── pages/
    ├── 1_Home.py              # Project overview & dataset info
    ├── 2_Predictor.py         # Startup prediction form
    ├── 3_AI_Insights.py       # Feature-driven strengths / risks
    ├── 4_What_If_Simulator.py # Compare baseline vs modified profile
    ├── 5_Analytics_Dashboard.py # EDA dashboards
    └── 6_Model_Performance.py # Metrics, ROC, confusion matrix
```

---

## ⚡ Quick Start

### 1. Clone / download the project

```bash
git clone https://github.com/yourusername/venturevision-ai.git
cd venturevision-ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model (one-time setup)

```bash
python train_model.py
```

This will:
- Load and preprocess 100K startup records
- Train 4 models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting)
- Select the best model by macro F1 score
- Save all artefacts to `models/`

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1 (macro) |
|---|---|---|---|---|
| Logistic Regression | 0.684 | 0.542 | 0.720 | 0.561 |
| **Random Forest** ⭐ | **0.732** | **0.622** | **0.618** | **0.620** |
| XGBoost | 0.743 | 0.696 | 0.577 | 0.610 |
| Gradient Boosting | 0.742 | 0.685 | 0.587 | 0.618 |

---

## 🎯 Target Classes

| Class | Description | Distribution |
|---|---|---|
| **Failure** | Company shut down | ~55.6% |
| **Acquisition** | Acquired by larger company | ~42.3% |
| **IPO** | Went public | ~2.1% |

---

## 🧩 Features Used

| Feature | Type | Description |
|---|---|---|
| `funding_rounds` | Numeric | Number of funding rounds |
| `founder_experience_years` | Numeric | Years of experience |
| `team_size` | Numeric | Total headcount |
| `market_size_billion` | Numeric | Market size in $B |
| `product_traction_users` | Numeric | Active user base |
| `burn_rate_million` | Numeric | Monthly cash burn ($M) |
| `revenue_million` | Numeric | Annual revenue ($M) |
| `investor_type` | Categorical | none / angel / tier2_vc / tier1_vc |
| `sector` | Categorical | AI / Climate / Crypto / Ecommerce / Fintech / Health / SaaS |
| `founder_background` | Categorical | first_time / academic / ex_bigtech / serial_founder |

---

## 📱 App Pages

| Page | Description |
|---|---|
| 🏠 **Home** | Overview, business problem, dataset info |
| 🔮 **Predictor** | Predict outcome with probability breakdown |
| 💡 **AI Insights** | Feature-importance-based strengths, risks, improvements |
| 🧪 **What-If Simulator** | Compare baseline vs modified scenario |
| 📈 **Analytics Dashboard** | 8+ interactive EDA charts with sector/investor filters |
| 🏆 **Model Performance** | Accuracy, confusion matrix, ROC curves, classification report |

---

## 🚀 Deployment

### Local

```bash
streamlit run app.py --server.port 8501
```

### Streamlit Community Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set `app.py` as entry point
4. Add dataset under `data/` and run `train_model.py` via a startup script or pre-commit the model artefacts

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
RUN python train_model.py
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
```

```bash
docker build -t venturevision-ai .
docker run -p 8501:8501 venturevision-ai
```

### Heroku / Railway / Render

Add a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

---

## 🛠️ Development Notes

- **Caching**: `@st.cache_resource` for model/transformer loading; `@st.cache_data` for dataset loading
- **Session state**: Predictor outputs flow to What-If and Insights pages via `st.session_state`
- **Modular architecture**: EDA, preprocessing, and model utilities are fully separated in `utils/`
- **Error handling**: All pages gracefully handle missing artefacts and dataset files

---

## 📄 License

MIT License – feel free to use, modify, and distribute.

---

*Built with ❤️ using Streamlit · Scikit-learn · XGBoost · Plotly*
# updated
