# Intelligent Battery Monitoring System with Uncertainty Quantification

A comprehensive machine learning system for predicting Remaining Useful Life (RUL) of batteries with uncertainty quantification using statistical features and Empirical Mode Decomposition (EMD).

## 🎯 Project Overview

I built an intelligent battery monitoring system that predicts battery RUL with confidence intervals using:
- **Statistical Features**: Voltage, current, temperature statistics from cycle waveforms (16 features)
- **EMD Features**: Empirical Mode Decomposition to capture multi-scale temporal patterns (159 features)
- **Multiple Models**: Random Forest (GridSearchCV optimized), LSTM (Optuna optimized), and Transformer
- **Uncertainty Quantification**: Monte Carlo Dropout for LSTM using PyTorch
- **Total Features**: 175 features per cycle (16 statistical + 159 EMD features)
- **Interactive Dashboard**: Streamlit web application for real-time predictions

## 📊 Dataset

I used NASA/CALCE battery datasets with:
- 2,750 discharge cycles
- 34 unique batteries
- Multi-scale temporal features extracted via EMD

## 🚀 Project Structure

### Phase 1: Point Prediction Models with Hyperparameter Optimization

I trained and optimized three models for point predictions:

1. **Random Forest** - Optimized with GridSearchCV
   - Test MAE: 18.82 cycles
   - Test RMSE: 23.61 cycles
   - Test R²: 0.244

2. **LSTM** - Optimized with Optuna (Bayesian Optimization)
   - Test MAE: 14.72 cycles
   - Test RMSE: 19.77 cycles
   - Test R²: 0.206

3. **Transformer** - Point prediction model
   - Test MAE: 19.06 cycles
   - Test RMSE: 23.58 cycles

### Phase 2: Uncertainty Quantification (LSTM)

I added Monte Carlo Dropout to the optimized LSTM model:
- 100 forward passes with dropout enabled at inference
- Mean predictions, standard deviation, and prediction intervals
- Test R² with MC Dropout: 0.426 (improved from 0.206)
- Better uncertainty calibration with optimized hyperparameters

## 📁 Project Structure

```
Battery_RUL/
├── src/
│   ├── features/          # Feature extraction modules
│   │   ├── emd_extractor.py
│   │   └── feature_pipeline.py
│   └── visualization/     # Plotting utilities
├── notebooks/
│   ├── exploration/       # Data exploration notebooks
│   ├── modeling/          # Model training notebooks
│   │   ├── 01_extract_emd_features.ipynb          ✅ EMD feature extraction
│   │   ├── 02_train_random_forest_point.ipynb    ✅ Random Forest (GridSearchCV)
│   │   ├── 03_train_lstm_pytorch.ipynb           ✅ LSTM (Optuna optimized)
│   │   ├── 04_train_transformer_point.ipynb       ✅ Transformer
│   │   ├── 05_compare_models_point.ipynb         ✅ Model comparison
│   │   ├── 06_add_uncertainty_lstm_mc_pytorch.ipynb ✅ MC Dropout
│   │   └── 09_compare_optimized_models.ipynb      ✅ Optimized models comparison
│   └── evaluation/        # Visualization notebooks
├── data/
│   ├── processed/         # Processed datasets
│   │   └── rul_features_with_emd.parquet
│   └── raw/              # Raw data
├── results/
│   ├── models/           # Saved models (.pkl, .pth, .json)
│   └── visualizations/   # Generated plots
├── app.py                # Streamlit dashboard
└── docs/                 # Documentation
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

### Training Models

#### 1. Extract EMD Features
```bash
jupyter notebook notebooks/modeling/01_extract_emd_features.ipynb
```

#### 2. Train Random Forest Model (GridSearchCV Optimized)
```bash
jupyter notebook notebooks/modeling/02_train_random_forest_point.ipynb
```
This notebook uses GridSearchCV to find the best hyperparameters automatically.

#### 3. Train LSTM Model (Optuna Optimized)
```bash
jupyter notebook notebooks/modeling/03_train_lstm_pytorch.ipynb
```
This notebook uses Optuna for Bayesian hyperparameter optimization. Uses PyTorch with MPS acceleration for Apple Silicon.

#### 4. Train Transformer Model
```bash
jupyter notebook notebooks/modeling/04_train_transformer_point.ipynb
```

#### 5. Compare All Models
```bash
jupyter notebook notebooks/modeling/05_compare_models_point.ipynb
```

#### 6. Add Monte Carlo Dropout to LSTM
```bash
jupyter notebook notebooks/modeling/06_add_uncertainty_lstm_mc_pytorch.ipynb
```

#### 7. Compare Optimized Models
```bash
jupyter notebook notebooks/modeling/09_compare_optimized_models.ipynb
```

### Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## 📈 Results

### Model Performance Summary

| Model | Optimization Method | Test MAE | Test RMSE | Test R² |
|-------|-------------------|----------|-----------|---------|
| **Random Forest** | GridSearchCV | 18.82 cycles | 23.61 cycles | 0.244 |
| **LSTM** | Optuna (Bayesian) | 14.72 cycles | 19.77 cycles | 0.206 |
| **LSTM (MC Dropout)** | Optuna + MC Dropout | - | - | **0.426** |

**Key Findings:**
- LSTM achieves the best MAE (14.72 cycles) - 22% better than Random Forest
- Random Forest achieves the best R² (0.244) for point predictions
- MC Dropout significantly improves LSTM performance (R²: 0.206 → 0.426)
- Both models benefit significantly from hyperparameter optimization

### Hyperparameters Found

**Random Forest (GridSearchCV):**
- n_estimators: 50
- max_depth: 10
- min_samples_split: 2
- min_samples_leaf: 4
- max_features: 'log2'

**LSTM (Optuna):**
- hidden_size1: 112
- hidden_size2: 32
- dropout: 0.1
- learning_rate: 0.0023
- batch_size: 64

## 🔬 Methodology

### Data Preprocessing Pipeline

1. **Metadata Processing**:
   - Load battery metadata with capacity measurements
   - Filter discharge cycles with valid capacity (> 0)
   - Calculate cycle index per battery

2. **RUL Label Generation**:
   - Compute initial capacity per battery
   - Calculate SOH (State of Health) = Current Capacity / Initial Capacity
   - Identify EOL cycle (first cycle where SOH ≤ 0.8)
   - Calculate RUL = EOL_cycle - cycle_index

3. **Feature Extraction**:
   - **Statistical Features** (16 features): Voltage, current, temperature statistics
   - **EMD Features** (159 features): Empirical Mode Decomposition of voltage, current, temperature signals
   - Total: 175 features per cycle

4. **Data Splitting**:
   - Battery-level splits (70% train, 15% val, 15% test)
   - Prevents data leakage by ensuring same battery doesn't appear in multiple splits

### Hyperparameter Optimization

**Random Forest**: I used GridSearchCV with 5-fold cross-validation to search through 405 parameter combinations, optimizing for MAE.

**LSTM**: I used Optuna (Bayesian Optimization) with 20 trials, using MedianPruner for early stopping. This automatically found the best hyperparameters without exhaustive search.

### Uncertainty Quantification

**Monte Carlo Dropout**: 
- Run 100 forward passes with dropout enabled during inference
- Extract mean and standard deviation from predictions
- Calculate prediction intervals (5th, 25th, 75th, 95th percentiles)
- Evaluate uncertainty calibration

## 🎨 Dashboard

The Streamlit dashboard (`app.py`) provides:
- **Model Selection**: Choose from Random Forest, LSTM (with Uncertainty), or Transformer
- **Interactive Predictions**: Select battery and cycle from test data
- **Uncertainty Visualization**: See confidence intervals for LSTM predictions
- **Model Comparison**: Compare all models side-by-side
- **Real-time Results**: Instant predictions with visualizations

## 📚 Documentation

- [Dashboard Guide](README_DASHBOARD.md) - How to use the dashboard
- [Dashboard Explanation](DASHBOARD_EXPLANATION.md) - What the dashboard demonstrates
- [Project Roadmap](docs/PROJECT_ROADMAP.md) - Complete project structure

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
- Optuna for hyperparameter optimization

---

**Status**: 
- ✅ **Phase 1**: All models trained with hyperparameter optimization
- ✅ **Phase 2**: MC Dropout implemented and evaluated
- ✅ **Dashboard**: Interactive web interface complete
- ✅ **Optimization**: GridSearchCV (RF) and Optuna (LSTM) implemented
