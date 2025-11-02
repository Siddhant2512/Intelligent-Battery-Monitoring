# Intelligent Battery Monitoring System with Uncertainty Quantification

A comprehensive machine learning system for predicting Remaining Useful Life (RUL) of batteries with uncertainty quantification using statistical features and Empirical Mode Decomposition (EMD).

## 🎯 Project Overview

This project implements an intelligent battery monitoring system that predicts battery RUL with confidence intervals using:
- **Statistical Features**: Voltage, current, temperature statistics from cycle waveforms
- **EMD Features**: Empirical Mode Decomposition to capture multi-scale temporal patterns
- **Multiple Models**: Random Forest, LSTM with Monte Carlo Dropout, and Transformer
- **Uncertainty Quantification**: Prediction intervals and confidence measures
- **Web Dashboard**: Interactive visualization of predictions (coming soon)

## 📊 Dataset

Uses NASA/CALCE battery datasets with:
- 2,750 discharge cycles
- 34 unique batteries
- Multi-scale temporal features extracted via EMD

## 🚀 Features

- **Feature Extraction**:
  - Basic statistical features (16 features per cycle)
  - EMD decomposition features (159 features per cycle)
  - Total: 175 features per cycle

- **Models**:
  - Random Forest with ensemble uncertainty quantification
  - LSTM with Monte Carlo Dropout
  - Simple Transformer model

- **Uncertainty Quantification**:
  - 90% and 50% prediction intervals
  - Calibration metrics
  - Confidence interval visualization

## 📁 Project Structure

```
Battery_RUL/
├── src/
│   ├── features/          # Feature extraction modules
│   │   ├── emd_extractor.py
│   │   └── feature_pipeline.py
│   ├── models/            # Model implementations
│   └── visualization/     # Plotting utilities
├── notebooks/
│   ├── exploration/       # Data exploration notebooks
│   └── modeling/          # Model training notebooks
├── data/
│   ├── processed/         # Processed datasets
│   ├── raw/              # Raw data
│   └── external/         # External data sources
├── results/
│   ├── models/           # Saved models
│   ├── figures/          # Generated plots
│   └── reports/          # Evaluation reports
├── dashboard/            # Web dashboard (coming soon)
└── docs/                # Documentation

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

### 1. Extract EMD Features
```bash
jupyter notebook notebooks/modeling/01_extract_emd_features.ipynb
```

### 2. Train Random Forest Model
```bash
jupyter notebook notebooks/modeling/02_train_random_forest.ipynb
```

### 3. Train LSTM Model (Coming Soon)
```bash
jupyter notebook notebooks/modeling/03_train_lstm_mc.ipynb
```

### 4. Train Transformer Model (Coming Soon)
```bash
jupyter notebook notebooks/modeling/04_train_transformer.ipynb
```

## 📈 Results

- **Random Forest**: Baseline model with ensemble uncertainty
- **LSTM**: Deep learning model with Monte Carlo Dropout
- **Transformer**: Attention-based model for sequence prediction

## 🔬 Methodology

1. **Feature Extraction**:
   - Statistical features from voltage, current, temperature signals
   - EMD decomposition to extract IMFs (Intrinsic Mode Functions)
   - Feature engineering from IMF statistics

2. **Uncertainty Quantification**:
   - Random Forest: Ensemble of models with different random seeds
   - LSTM: Monte Carlo Dropout (100 forward passes at inference)
   - Transformer: Quantile regression or ensemble methods

3. **Model Evaluation**:
   - MAE, RMSE, MAPE, R² metrics
   - Prediction interval coverage (calibration)
   - Visualizations with confidence intervals

## 🎨 Dashboard

Interactive web dashboard (coming soon) for:
- Real-time RUL predictions
- Confidence interval visualization
- Battery health monitoring
- Historical trend analysis

## 📚 Documentation

- [Project Roadmap](docs/PROJECT_ROADMAP.md)
- [Modeling Sequence Guide](docs/MODELING_SEQUENCE.md)
- [Next Steps](docs/NEXT_STEPS.md)

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

**Status**: 🚧 In Development - Core features implemented, dashboard coming soon

