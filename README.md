# Intelligent Battery Monitoring System with Uncertainty Quantification

A comprehensive machine learning system for predicting Remaining Useful Life (RUL) of batteries with uncertainty quantification using statistical features and Empirical Mode Decomposition (EMD).

## 🎯 Project Overview

This project implements an intelligent battery monitoring system that predicts battery RUL with confidence intervals using:
- **Statistical Features**: Voltage, current, temperature statistics from cycle waveforms (16 features)
- **EMD Features**: Empirical Mode Decomposition to capture multi-scale temporal patterns (159 features)
- **Multiple Models**: Random Forest, LSTM, and Transformer for point predictions
- **Uncertainty Quantification**: Monte Carlo Dropout for LSTM (Phase 2)
- **Total Features**: 175 features per cycle

## 📊 Dataset

Uses NASA/CALCE battery datasets with:
- 2,750 discharge cycles
- 34 unique batteries
- Multi-scale temporal features extracted via EMD

## 🚀 Project Structure

### Phase 1: Point Prediction Models (Baseline Comparison)
Compare Random Forest, LSTM, and Transformer models on point predictions:
1. **Random Forest** - Point prediction with feature importance
2. **LSTM** - Sequence-based point prediction
3. **Transformer** - Attention-based point prediction
4. **Model Comparison** - Compare all 3 models on test set

### Phase 2: Uncertainty Quantification (LSTM Only)
Add Monte Carlo Dropout to LSTM model:
- 100 forward passes with dropout enabled at inference
- Extract mean and std from predictions
- Calculate prediction intervals (5th, 25th, 75th, 95th percentiles)
- Evaluate uncertainty calibration

## 📁 Project Structure

```
Battery_RUL/
├── src/
│   ├── features/          # Feature extraction modules
│   │   ├── emd_extractor.py
│   │   └── feature_pipeline.py
│   ├── models/            # Model implementations (to be created)
│   └── visualization/     # Plotting utilities (to be created)
├── notebooks/
│   ├── exploration/       # Data exploration notebooks
│   └── modeling/          # Model training notebooks
│       ├── 01_extract_emd_features.ipynb          ✅ Done
│       ├── 02_train_random_forest_point.ipynb      ✅ Point prediction
│       ├── 03_train_lstm_point.ipynb               ← Point prediction
│       ├── 04_train_transformer_point.ipynb         ← Point prediction
│       ├── 05_compare_models_point.ipynb           ← Compare all 3
│       └── 06_add_uncertainty_lstm_mc.ipynb         ← MC Dropout for LSTM
├── data/
│   ├── processed/         # Processed datasets
│   │   ├── rul_features.csv
│   │   └── rul_features_with_emd.parquet
│   ├── raw/              # Raw data
│   └── external/         # External data sources
├── results/
│   ├── models/           # Saved models
│   ├── figures/          # Generated plots
│   └── reports/          # Evaluation reports
├── dashboard/            # Web dashboard (coming soon)
└── docs/                 # Documentation
    ├── PROJECT_ROADMAP.md
    ├── MODELING_SEQUENCE.md
    └── NEXT_STEPS.md
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/Siddhant2512/Intelligent-Battery-Monitoring.git
cd Intelligent-Battery-Monitoring
```

2. Create virtual environment:
```bash
python -m venv battery_env
source battery_env/bin/activate  # On Windows: battery_env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📝 Usage

### Phase 1: Point Prediction Models

#### 1. Extract EMD Features
```bash
jupyter notebook notebooks/modeling/01_extract_emd_features.ipynb
```

#### 2. Train Random Forest Model
```bash
jupyter notebook notebooks/modeling/02_train_random_forest_point.ipynb
```

#### 3. Train LSTM Model
```bash
jupyter notebook notebooks/modeling/03_train_lstm_point.ipynb
```

#### 4. Train Transformer Model
```bash
jupyter notebook notebooks/modeling/04_train_transformer_point.ipynb
```

#### 5. Compare All Models
```bash
jupyter notebook notebooks/modeling/05_compare_models_point.ipynb
```

### Phase 2: Uncertainty Quantification (LSTM Only)

#### 6. Add Monte Carlo Dropout to LSTM
```bash
jupyter notebook notebooks/modeling/06_add_uncertainty_lstm_mc.ipynb
```

## 📈 Results

### Phase 1: Point Predictions
- **Random Forest**: Baseline model with feature importance
- **LSTM**: Deep learning model with sequence learning
- **Transformer**: Attention-based model for sequence prediction

### Phase 2: Uncertainty (LSTM)
- **Monte Carlo Dropout**: 100 forward passes with dropout
- Prediction intervals with calibration metrics
- Confidence interval visualization

## 🔬 Methodology

### Feature Extraction
1. **Statistical Features** (16 features):
   - Voltage, current, temperature statistics
   - Duration, coulomb count, IR drop

2. **EMD Features** (159 features):
   - Empirical Mode Decomposition of voltage, current, temperature signals
   - IMF (Intrinsic Mode Functions) statistics: energy, mean, std, skewness, kurtosis
   - Cross-IMF correlations

### Uncertainty Quantification (LSTM Only)
- **Monte Carlo Dropout**: 
  - Enable dropout layers during inference
  - Run 100 forward passes
  - Extract mean and standard deviation
  - Calculate prediction intervals

### Model Evaluation
- **Point Prediction Metrics**: MAE, RMSE, MAPE, R²
- **Uncertainty Metrics** (LSTM): Prediction Interval Coverage, Calibration curves
- **Visualizations**: Predictions vs actual with confidence intervals

## 🎨 Dashboard (Future)

Interactive web dashboard (coming soon) for:
- Real-time RUL predictions
- Confidence interval visualization (LSTM)
- Battery health monitoring
- Historical trend analysis

## 📚 Documentation

- [Project Roadmap](docs/PROJECT_ROADMAP.md) - Complete project structure and phases
- [Modeling Sequence Guide](docs/MODELING_SEQUENCE.md) - Step-by-step modeling guide
- [Next Steps](docs/NEXT_STEPS.md) - Quick start guide

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Siddhant Aggarwal**

- GitHub: [@Siddhant2512](https://github.com/Siddhant2512)
- Repository: [Intelligent-Battery-Monitoring](https://github.com/Siddhant2512/Intelligent-Battery-Monitoring)

## 🙏 Acknowledgments

- NASA Battery Dataset
- CALCE Battery Research Group
- PyEMD library for Empirical Mode Decomposition

---

**Status**: 🚧 In Development - Phase 1 in progress, Phase 2 (MC Dropout) coming next
