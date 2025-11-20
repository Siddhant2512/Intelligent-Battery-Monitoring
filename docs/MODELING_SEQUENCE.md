# Modeling Sequence Guide

## ✅ Completed Steps

1. **Feature Extraction** - Basic stats + EMD features extracted (`01_extract_emd_features.ipynb`)
2. **Random Forest** - Trained with GridSearchCV optimization (`02_train_random_forest_point.ipynb`)
3. **LSTM** - Trained with Optuna optimization (`03_train_lstm_pytorch.ipynb`)
4. **Transformer** - Trained for point predictions (`04_train_transformer_point.ipynb`)
5. **Model Comparison** - Compared all models (`05_compare_models_point.ipynb`)
6. **MC Dropout** - Added uncertainty to LSTM (`06_add_uncertainty_lstm_mc_pytorch.ipynb`)
7. **Optimized Comparison** - Compared optimized models (`09_compare_optimized_models.ipynb`)
8. **Dashboard** - Interactive web interface (`app.py`)

## 🚀 Workflow (In Order)

### Step 1: Extract EMD Features ✅
**Notebook:** `notebooks/modeling/01_extract_emd_features.ipynb`

**What it does:**
- Extracts Empirical Mode Decomposition features from voltage, current, and temperature signals
- Adds 159 EMD features to the 16 statistical features
- Total: 175 features per cycle

**Expected runtime:** 30-60 minutes

**Outputs:**
- `data/processed/rul_features_with_emd.parquet`

---

### Step 2: Train Random Forest with GridSearchCV ✅
**Notebook:** `notebooks/modeling/02_train_random_forest_point.ipynb`

**What it does:**
- Trains Random Forest for RUL prediction
- Uses GridSearchCV to find best hyperparameters automatically
- Searches through 405 parameter combinations
- Evaluates with 5-fold cross-validation
- Saves optimized model and predictions

**Expected runtime:** 5-15 minutes

**Outputs:**
- `results/models/random_forest_rul_point_model.pkl`
- `results/models/rf_predictions_point.csv`
- `results/models/rf_metrics_point.csv`

**Best Parameters Found:**
- n_estimators: 50
- max_depth: 10
- min_samples_split: 2
- min_samples_leaf: 4
- max_features: 'log2'

**Performance:**
- Test MAE: 18.82 cycles
- Test RMSE: 23.61 cycles
- Test R²: 0.244

---

### Step 3: Train LSTM with Optuna Optimization ✅
**Notebook:** `notebooks/modeling/03_train_lstm_pytorch.ipynb`

**What it does:**
- Prepares sequences (past 20 cycles → RUL)
- Builds LSTM model with dropout layers
- Uses Optuna for Bayesian hyperparameter optimization
- Runs 20 trials with MedianPruner for early stopping
- Trains final model with best hyperparameters

**Expected runtime:** 30-60 minutes (depending on Optuna trials)

**Outputs:**
- `results/models/lstm_pytorch_point_model.pth`
- `results/models/lstm_pytorch_model_info.json`
- `results/models/lstm_pytorch_scaler.pkl`
- `results/models/lstm_pytorch_predictions_point.csv`
- `results/models/lstm_pytorch_metrics_point.csv`

**Best Parameters Found:**
- hidden_size1: 112
- hidden_size2: 32
- dropout: 0.1
- learning_rate: 0.0023
- batch_size: 64

**Performance:**
- Test MAE: 14.72 cycles
- Test RMSE: 19.77 cycles
- Test R²: 0.206

---

### Step 4: Train Transformer ✅
**Notebook:** `notebooks/modeling/04_train_transformer_point.ipynb`

**What it does:**
- Uses positional encoding for cycle sequences
- Multi-head self-attention mechanism
- Trains Transformer model for point predictions

**Expected runtime:** 45-90 minutes

**Performance:**
- Test MAE: 19.06 cycles
- Test RMSE: 23.58 cycles

---

### Step 5: Compare All Models ✅
**Notebook:** `notebooks/modeling/05_compare_models_point.ipynb`

**What it does:**
- Compares all three models on same test set
- Metrics: MAE, RMSE, R²
- Visualizations comparing all models
- Identifies best performing model

**Expected runtime:** 10-20 minutes

---

### Step 6: Add Monte Carlo Dropout to LSTM ✅
**Notebook:** `notebooks/modeling/06_add_uncertainty_lstm_mc_pytorch.ipynb`

**What it does:**
- Uses optimized LSTM model (Optuna hyperparameters)
- At inference: 100 forward passes with dropout enabled
- Extracts mean and std from predictions
- Calculates prediction intervals (5th, 25th, 75th, 95th percentiles)
- Evaluates uncertainty calibration metrics

**Expected runtime:** 5-10 minutes

**Performance:**
- Test R²: 0.426 (improved from 0.206)
- Better uncertainty calibration with optimized model

---

### Step 7: Compare Optimized Models ✅
**Notebook:** `notebooks/modeling/09_compare_optimized_models.ipynb`

**What it does:**
- Compares Random Forest (GridSearchCV) vs LSTM (Optuna)
- Shows improvement from hyperparameter optimization
- Visualizes performance differences
- Provides recommendations

**Expected runtime:** 5 minutes

---

### Step 8: Dashboard Deployment ✅
**File:** `app.py`

**What it does:**
- Streamlit web application
- Real-time RUL predictions
- Confidence intervals visualization (LSTM)
- Model comparison interface
- Interactive battery health monitoring

**Running:**
```bash
streamlit run app.py
```

---

## Quick Start

1. **Run notebooks in order:**
   ```bash
   # 1. Extract features
   jupyter notebook notebooks/modeling/01_extract_emd_features.ipynb
   
   # 2. Train Random Forest (GridSearchCV)
   jupyter notebook notebooks/modeling/02_train_random_forest_point.ipynb
   
   # 3. Train LSTM (Optuna)
   jupyter notebook notebooks/modeling/03_train_lstm_pytorch.ipynb
   
   # 4. Add MC Dropout
   jupyter notebook notebooks/modeling/06_add_uncertainty_lstm_mc_pytorch.ipynb
   ```

2. **Run dashboard:**
   ```bash
   streamlit run app.py
   ```

## Tips

- **Hyperparameter Optimization**: GridSearchCV for RF, Optuna for LSTM - both significantly improve performance
- **Sequence length**: 20 cycles works well for LSTM/Transformer
- **MC Dropout**: 100 forward passes provides stable uncertainty estimates
- **Model Selection**: LSTM for best accuracy, Random Forest for speed and interpretability

---

**All steps completed! The system is ready for deployment. 🎯**
