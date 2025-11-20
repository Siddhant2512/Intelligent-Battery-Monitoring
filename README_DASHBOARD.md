# Battery RUL Prediction Dashboard

A Streamlit web application for predicting battery Remaining Useful Life (RUL) with uncertainty quantification.

## Features

- **Multiple Models**: Random Forest (GridSearchCV optimized), LSTM (Optuna optimized with uncertainty), and Transformer
- **Uncertainty Quantification**: Monte Carlo Dropout for LSTM predictions
- **Model Comparison**: Side-by-side comparison of all models
- **Interactive Interface**: Easy selection from test data

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure all models are trained and saved in `results/models/`:
   - `random_forest_rul_point_model.pkl` (GridSearchCV optimized)
   - `lstm_pytorch_point_model.pth` (Optuna optimized)
   - `lstm_pytorch_model_info.json`
   - `lstm_pytorch_scaler.pkl`
   - `transformer_point_model.pth`
   - `transformer_model_info.json`
   - `transformer_scaler.pkl`

## Running the Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Usage

1. **Select Model**: Choose from Random Forest, LSTM (with Uncertainty), or Transformer
2. **Input Data**: Select battery ID and cycle from test data
3. **View Predictions**: See RUL predictions with uncertainty intervals (for LSTM)
4. **Compare Models**: Use "Compare All" to see predictions from all models side-by-side

## Model Performance

### Optimized Models (Current)

**Random Forest (GridSearchCV):**
- Test MAE: 18.82 cycles
- Test RMSE: 23.61 cycles
- Test R²: 0.244
- Best for: Interpretability and speed

**LSTM (Optuna + MC Dropout):**
- Test MAE: 14.72 cycles (point prediction)
- Test RMSE: 19.77 cycles (point prediction)
- Test R²: 0.206 (point), **0.426 (MC Dropout)**
- Best for: Accuracy and uncertainty quantification

**Transformer:**
- Test MAE: 19.06 cycles
- Test RMSE: 23.58 cycles
- Best for: Alternative deep learning approach

## Model Requirements

### Random Forest
- Requires: Feature vector (175 features)
- Output: Point prediction
- Uses: Full feature set with GridSearchCV optimized hyperparameters

### LSTM
- Requires: Sequence of 20 cycles (175 features each)
- Output: Point prediction + uncertainty (via MC Dropout)
- Uncertainty: 90% confidence intervals
- Uses: Optuna optimized hyperparameters (hidden_size1=112, hidden_size2=32, dropout=0.1)

### Transformer
- Requires: Sequence of 20 cycles (175 features each)
- Output: Point prediction

## Notes

- LSTM and Transformer require sequences of 20 cycles, so the selected battery must have at least 20 cycles
- MC Dropout samples can be adjusted in the sidebar (default: 100)
- All models use the full feature set (175 features: 16 statistical + 159 EMD)
- Models are optimized with hyperparameter tuning (GridSearchCV for RF, Optuna for LSTM)

## Troubleshooting

If you see "Model not loaded" errors:
1. Ensure all model files exist in `results/models/`
2. Check that models were trained successfully
3. Restart Streamlit: `streamlit cache clear` then restart
4. Check the error message in the dashboard for specific file issues
