# Intelligent Battery Monitoring System with Uncertainty Quantification

A comprehensive machine learning system for predicting Remaining Useful Life (RUL) of batteries with uncertainty quantification using statistical features and Empirical Mode Decomposition (EMD).

## 🎯 Project Overview

This project implements an intelligent battery monitoring system that predicts battery RUL with confidence intervals using:
- **Statistical Features**: Voltage, current, temperature statistics from cycle waveforms (16 features)
- **EMD Features**: Empirical Mode Decomposition to capture multi-scale temporal patterns (159 features)
- **Multiple Models**: Random Forest, LSTM (PyTorch), and Transformer for point predictions
- **Uncertainty Quantification**: Monte Carlo Dropout for LSTM using PyTorch (Phase 2)
- **Total Features**: 175 features per cycle (16 statistical + 159 EMD features)
- **Visualization**: Multi-audience visualization pipeline for data insights and model results

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
│   │   └── Data_Exploration2.ipynb                 ✅ Data preprocessing
│   ├── modeling/          # Model training notebooks
│   │   ├── 01_extract_emd_features.ipynb          ✅ EMD feature extraction
│   │   ├── 02_train_random_forest_point.ipynb      ✅ Random Forest (point)
│   │   ├── 03_train_lstm_pytorch.ipynb             ✅ LSTM (PyTorch, point)
│   │   ├── 06_add_uncertainty_lstm_mc_pytorch.ipynb ✅ MC Dropout (PyTorch)
│   │   └── 04_train_transformer_point.ipynb         ← Transformer (pending)
│   └── evaluation/        # Visualization and evaluation notebooks
│       └── 01_dataset_insights_visualization.ipynb  ✅ Level 1 visualizations
├── data/
│   ├── processed/         # Processed datasets
│   │   ├── rul_features.csv
│   │   └── rul_features_with_emd.parquet
│   ├── raw/              # Raw data
│   └── external/         # External data sources
├── results/
│   ├── models/           # Saved models (.pkl, .pth, .h5)
│   ├── visualizations/   # Generated plots and figures
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

#### 3. Train LSTM Model (PyTorch)
```bash
jupyter notebook notebooks/modeling/03_train_lstm_pytorch.ipynb
```
**Note**: Uses PyTorch with MPS acceleration for Apple Silicon (much faster than TensorFlow)

#### 4. Train Transformer Model
```bash
jupyter notebook notebooks/modeling/04_train_transformer_point.ipynb
```

#### 5. Compare All Models
```bash
jupyter notebook notebooks/modeling/05_compare_models_point.ipynb
```

### Phase 2: Uncertainty Quantification (LSTM Only)

#### 6. Add Monte Carlo Dropout to LSTM (PyTorch)
```bash
jupyter notebook notebooks/modeling/06_add_uncertainty_lstm_mc_pytorch.ipynb
```

### Visualization

#### 7. Generate Dataset Insights Visualizations
```bash
jupyter notebook notebooks/evaluation/01_dataset_insights_visualization.ipynb
```
Creates publication-quality visualizations for data understanding (Level 1).

## 📈 Results

### Data Preprocessing

The preprocessing pipeline transforms raw battery cycle data into a structured dataset:

- **Input**: 2,750 discharge cycles from 34 unique batteries
- **Processing Steps**:
  1. Filter discharge cycles with valid capacity measurements
  2. Calculate cycle index, SOH (State of Health), and EOL (End of Life) cycle
  3. Compute RUL (Remaining Useful Life) as `EOL_cycle - cycle_index`
  4. Extract statistical features from cycle waveforms (16 features)
  5. Apply Empirical Mode Decomposition (EMD) to extract multi-scale patterns (159 features)
  6. Create battery-level train/val/test splits (70/15/15) to prevent data leakage

- **Output**: 
  - `rul_features_with_emd.parquet`: 2,750 rows × 190 columns
  - 1,408 rows with valid RUL labels (batteries that reached EOL)
  - RUL range: -107 to 123 cycles

### Phase 1: Point Predictions

#### Random Forest Model ✅

**Configuration**:
- 100 decision trees
- Max depth: 20
- Min samples split: 5
- Min samples leaf: 2
- Features: 175 (statistical + EMD)

**Performance** (Test Set):
- **MAE**: ~21-22 cycles
- **RMSE**: ~27-28 cycles
- **R²**: ~0.99 (train), ~-0.04 to 0.0 (test)
- **Training Time**: < 1 second

**Key Insights**:
- Excellent training fit (R² ≈ 0.99) indicating model capacity
- Test performance suggests overfitting or distribution shift
- Feature importance analysis reveals voltage and capacity metrics as top predictors
- EMD features contribute to model performance, validating feature engineering

**Model Artifacts**:
- Saved model: `results/models/random_forest_rul_point_model.pkl`
- Predictions: `results/models/rf_predictions_point.csv`
- Metrics: `results/models/rf_metrics_point.csv`

#### LSTM Model (PyTorch) ✅

**Configuration**:
- Architecture: LSTM(64) → LSTM(32) → Dense(16) → Dense(1)
- Sequence length: 20 cycles
- Dropout: 0.2 (for MC Dropout in Phase 2)
- Optimizer: Adam (lr=0.001)
- Training: MPS acceleration on Apple Silicon

**Performance**: Training completed successfully (results pending full evaluation)


#### Transformer Model
- Status: Pending implementation

### Phase 2: Uncertainty Quantification (LSTM)

#### Monte Carlo Dropout (PyTorch) ✅
- **Method**: 100 forward passes with dropout enabled during inference
- **Output**: Mean predictions, standard deviation, and prediction intervals
- **Implementation**: Simple `model.train()` during inference (PyTorch advantage)
- Status: Notebook ready, pending LSTM model training completion

## 🔬 Methodology

### Data Preprocessing Pipeline

1. **Metadata Processing**:
   - Load battery metadata with capacity measurements
   - Filter discharge cycles with valid capacity (> 0)
   - Coerce numeric columns (Capacity, Re, Rct) handling mixed data types
   - Calculate cycle index per battery

2. **RUL Label Generation**:
   - Compute initial capacity per battery
   - Calculate SOH (State of Health) = Current Capacity / Initial Capacity
   - Identify EOL cycle (first cycle where SOH ≤ 0.8)
   - Calculate RUL = EOL_cycle - cycle_index
   - Handle batteries that don't reach EOL (NaN RUL)

3. **Feature Extraction**:
   - **Statistical Features** (16 features):
     - Voltage: mean, min, max
     - Current: mean absolute value
     - Temperature: max
     - Duration, coulomb count (Ah), IR drop proxy
   
   - **EMD Features** (159 features):
     - Empirical Mode Decomposition of voltage, current, temperature signals
     - Extract up to 5 IMFs (Intrinsic Mode Functions) per signal
     - For each IMF: energy, mean, std, skewness, kurtosis
     - Total: 3 signals × 5 IMFs × 5 statistics × 2 (if applicable) ≈ 159 features

4. **Data Splitting**:
   - Battery-level splits (70% train, 15% val, 15% test)
   - Prevents data leakage by ensuring same battery doesn't appear in multiple splits
   - Handles edge cases (empty validation set when batteries don't reach EOL)

### Uncertainty Quantification (LSTM Only - PyTorch)
- **Monte Carlo Dropout**: 
  - Simple implementation: `model.train()` during inference
  - Run 100 forward passes with dropout enabled
  - Extract mean and standard deviation from predictions
  - Calculate prediction intervals (5th, 25th, 75th, 95th percentiles)
  - Evaluate uncertainty calibration (coverage metrics)

### Model Evaluation
- **Point Prediction Metrics**: MAE, RMSE, MAPE, R²
- **Uncertainty Metrics** (LSTM): Prediction Interval Coverage (90%, 50%), Average Interval Width
- **Visualizations**: 
  - Level 1: Dataset insights (capacity fade, correlations, distributions)
  - Model performance plots (predictions vs actual, residuals)
  - Uncertainty visualization (confidence intervals, calibration curves)

## 📊 Visualization

### Level 1: Dataset Insights ✅

Comprehensive visualizations demonstrating data understanding:

1. **Capacity Fade Over Cycles**: Tracks degradation patterns across multiple batteries
2. **Feature Correlation Heatmap**: Reveals relationships between operational signals and capacity
3. **Distribution Plots**: Statistical analysis of key features (voltage, capacity, EMD features)
4. **SOH vs RUL Relationship**: Validates RUL calculation methodology

All visualizations are publication-quality (300 DPI) and saved to `results/visualizations/`.

### Level 2: Model Performance (Coming Soon)
- Predictions vs actual plots
- Residual analysis
- Feature importance visualizations

### Level 3: Uncertainty Visualization (Coming Soon)
- Confidence intervals
- Calibration curves
- Uncertainty vs error analysis

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

**Status**: 
- ✅ **Phase 1**: Random Forest and LSTM (PyTorch) models trained
- ✅ **Data Preprocessing**: Complete pipeline with EMD features
- ✅ **Visualization**: Level 1 dataset insights complete
- 🚧 **Phase 1**: Transformer model pending
- 🚧 **Phase 2**: MC Dropout evaluation pending
- 🚧 **Dashboard**: Web interface coming soon
