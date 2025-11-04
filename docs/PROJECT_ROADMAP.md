# Intelligent Battery Monitoring System - Project Roadmap

## Project Overview
Build an intelligent battery monitoring system with uncertainty quantification for Remaining Useful Life (RUL) prediction using statistical features and Empirical Mode Decomposition (EMD).

## Current Status
✅ Basic statistical features extracted (16 features per cycle)  
✅ EMD features extracted (159 features per cycle)  
✅ Dataset processed and split (2,750 cycles, 34 batteries)  
✅ Total: 175 features per cycle

---

## Phase 1: Point Prediction Models (Baseline Comparison)
**Goal:** Train and compare Random Forest, LSTM, and Transformer models on point predictions only.

### 1.1 Random Forest - Point Prediction
- ✅ Train RF model for point predictions
- ✅ Evaluate: MAE, RMSE, R², MAPE
- ✅ Feature importance analysis
- ✅ Save baseline model

### 1.2 LSTM - Point Prediction  
- Prepare sequences (past N cycles → RUL)
- Train LSTM model (no dropout for point prediction)
- Evaluate: Same metrics as RF
- Save baseline model

### 1.3 Transformer - Point Prediction
- Prepare sequences with positional encoding
- Train Transformer model
- Evaluate: Same metrics as RF
- Save baseline model

### 1.4 Model Comparison
- Compare all 3 models on test set
- Create comparison plots (predictions vs actual)
- Feature importance/attention analysis
- Select best performing model

---

## Phase 2: Uncertainty Quantification (LSTM Only)
**Goal:** Add Monte Carlo Dropout to LSTM model for uncertainty quantification.

### 2.1 LSTM - Monte Carlo Dropout
- Use existing LSTM model or re-train with dropout enabled
- At inference: 100 forward passes with dropout enabled
- Extract mean and std from predictions
- Calculate prediction intervals (5th, 25th, 75th, 95th percentiles)
- Evaluate uncertainty calibration metrics
- Compare predictions with/without uncertainty

---

## Phase 3: Model Evaluation & Comparison
- **Point Prediction Metrics**: MAE, RMSE, MAPE, R² for all models
- **Uncertainty Metrics** (LSTM only): Prediction Interval Coverage, Calibration curves
- **Visualizations**: 
  - Prediction vs actual with confidence intervals (LSTM)
  - Side-by-side model comparison
  - Feature importance/attention analysis

---

## Phase 4: Web Dashboard (Future)
- **Backend**: Flask/FastAPI API
  - Model inference endpoints
  - Confidence interval generation (LSTM)
- **Frontend**: Interactive dashboard
  - Real-time RUL predictions
  - Confidence intervals visualization (LSTM)
  - Battery health monitoring charts

---

## Notebook Structure

```
notebooks/modeling/
  01_extract_emd_features.ipynb          ✅ Done - EMD feature extraction
  02_train_random_forest_point.ipynb      ✅ Point prediction only
  03_train_lstm_point.ipynb              ← Point prediction only  
  04_train_transformer_point.ipynb        ← Point prediction only
  05_compare_models_point.ipynb           ← Compare all 3 models
  06_add_uncertainty_lstm_mc.ipynb         ← MC Dropout for LSTM only
```

---

## File Structure

```
src/
  features/
    emd_extractor.py          ✅ EMD decomposition and feature extraction
    feature_pipeline.py       ✅ Complete feature extraction pipeline
    __init__.py               ✅ Module exports
  models/                     ← Model implementations (to be created)
    base_model.py             
    random_forest.py          
    lstm_mc.py                
    transformer.py            
  visualization/              ← Plotting utilities (to be created)
    uncertainty_plots.py       

notebooks/
  exploration/
    Data_Exploration1.ipynb   ✅ Initial data exploration
    Data_Exploration2.ipynb   ✅ Feature extraction and processing
  modeling/
    01_extract_emd_features.ipynb        ✅ Done
    02_train_random_forest_point.ipynb   ✅ Point prediction
    03_train_lstm_point.ipynb            ← To create
    04_train_transformer_point.ipynb      ← To create
    05_compare_models_point.ipynb         ← To create
    06_add_uncertainty_lstm_mc.ipynb       ← To create

dashboard/
  app.py                      ← Flask/FastAPI backend (future)
  static/                     ← Frontend assets (future)
  templates/                  ← HTML templates (future)

data/
  processed/
    rul_features.csv          ✅ Basic features
    rul_features_with_emd.parquet  ✅ With EMD features
  raw/
  external/

results/
  models/                     ← Saved models
  figures/                    ← Generated plots
  reports/                    ← Evaluation reports
```

---

## Benefits of This Approach

✅ **Systematic Comparison**: Compare models on equal footing (point predictions first)  
✅ **Clear Baseline**: Know which model performs best before adding complexity  
✅ **Focused Uncertainty**: Apply Monte Carlo Dropout only to LSTM (deep learning model)  
✅ **Better Analysis**: Can see if uncertainty improves predictions or adds overhead  
✅ **Simpler Implementation**: One uncertainty method (MC Dropout) for one model (LSTM)

---

## Next Steps

1. ✅ Extract EMD features (`01_extract_emd_features.ipynb`)
2. ✅ Train Random Forest point prediction (`02_train_random_forest_point.ipynb`)
3. ⏳ Create and train LSTM point prediction (`03_train_lstm_point.ipynb`)
4. ⏳ Create and train Transformer point prediction (`04_train_transformer_point.ipynb`)
5. ⏳ Compare all 3 models (`05_compare_models_point.ipynb`)
6. ⏳ Add Monte Carlo Dropout to LSTM (`06_add_uncertainty_lstm_mc.ipynb`)
7. ⏳ Build web dashboard (Phase 4)

---

## Dependencies

- **Feature Extraction**: PyEMD (EMD-signal), scipy, numpy, pandas
- **Models**: scikit-learn (Random Forest), tensorflow (LSTM, Transformer)
- **Visualization**: matplotlib, seaborn, plotly
- **Dashboard** (future): flask/fastapi

See `requirements.txt` for complete list.
