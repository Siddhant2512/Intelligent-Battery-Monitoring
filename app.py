"""
Intelligent Battery Monitoring System - Streamlit Dashboard

A simple web interface for RUL prediction with uncertainty quantification.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import sys

# Add project root to path
project_root = Path(__file__).parent.resolve()
sys.path.append(str(project_root))

# Page config
st.set_page_config(
    page_title="Battery RUL Prediction",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔋 Intelligent Battery Monitoring System</h1>', unsafe_allow_html=True)
st.markdown("### Remaining Useful Life (RUL) Prediction with Uncertainty Quantification")

# Initialize session state
def enable_dropout_during_inference(model):
    """
    Enable dropout layers while keeping BatchNorm/LayerNorm modules in eval mode.
    This avoids batch-norm errors when running Monte Carlo Dropout with batch size 1.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
    return model


def prepare_lstm_features(cycles_data, models, feature_cols):
    """
    Prepare features for LSTM prediction (optimized model uses full feature set).
    """
    # Optimized model uses all features (no PCA)
    seq_data = cycles_data[feature_cols].fillna(0).values.astype(np.float32)
    seq_reshaped = seq_data.reshape(-1, len(feature_cols))
    seq_scaled = models['lstm_scaler'].transform(seq_reshaped)
    return seq_scaled.reshape(seq_data.shape[0], -1, len(feature_cols))

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all trained models and scalers."""
    models_dir = project_root / "results" / "models"
    
    models = {}
    
    # Load Random Forest - use optimized model (GridSearchCV) with full features
    try:
        # Use optimized model (GridSearchCV): Test MAE 18.82, Test RMSE 23.61, Test R² 0.244
        models['rf'] = joblib.load(models_dir / "random_forest_rul_point_model.pkl")
        models['rf_use_reduced'] = False
        models['rf_loaded'] = True
        models['rf_optimized'] = True  # Mark as optimized with GridSearchCV
    except:
        models['rf_loaded'] = False
        models['rf_use_reduced'] = False
        models['rf_optimized'] = False
    
    # No PCA needed - using full feature set with optimized models
    models['pca'] = None
    models['pca_scaler'] = None
    
    # Load LSTM model info and create model
    # Use optimized model (Optuna Bayesian Optimization): Test MAE 14.72, Test RMSE 19.77, Test R² 0.206
    try:
        # Use optimized original LSTM (with Optuna)
        model_info_path = models_dir / "lstm_pytorch_model_info.json"
        model_path = models_dir / "lstm_pytorch_point_model.pth"
        scaler_path = models_dir / "lstm_pytorch_scaler.pkl"
        
        # Check if all required files exist
        if not model_info_path.exists():
            raise FileNotFoundError(f"LSTM model info not found: {model_info_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"LSTM model weights not found: {model_path}")
        if not scaler_path.exists():
            raise FileNotFoundError(f"LSTM scaler not found: {scaler_path}")
        
        with open(model_info_path, 'r') as f:
            lstm_info = json.load(f)
        
        # Verify we're reading the correct values
        print(f"DEBUG: Loaded LSTM info - hidden_size1={lstm_info.get('hidden_size1')}, hidden_size2={lstm_info.get('hidden_size2')}, dropout={lstm_info.get('dropout')}")
        
        # Define model classes
        class LSTMModel(nn.Module):
            """Original LSTM model architecture."""
            def __init__(self, input_size, hidden_size1=64, hidden_size2=32, num_layers=1, dropout=0.2):
                super(LSTMModel, self).__init__()
                self.lstm1 = nn.LSTM(input_size, hidden_size1, num_layers, 
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, num_layers,
                                    batch_first=True, dropout=dropout if num_layers > 1 else 0)
                self.fc1 = nn.Linear(hidden_size2, 16)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(16, 1)
            
            def forward(self, x):
                out, _ = self.lstm1(x)
                out, _ = self.lstm2(out)
                out = out[:, -1, :]
                out = self.fc1(out)
                out = self.relu(out)
                out = self.dropout(out)
                out = self.fc2(out)
                return out.squeeze()
        
        class ImprovedLSTMModel(nn.Module):
            """Improved LSTM model with attention and batch normalization."""
            def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.3, bidirectional=True):
                super(ImprovedLSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                self.bidirectional = bidirectional
                
                # Bidirectional LSTM
                self.lstm = nn.LSTM(
                    input_size, 
                    hidden_size, 
                    num_layers, 
                    batch_first=True, 
                    dropout=dropout if num_layers > 1 else 0,
                    bidirectional=bidirectional
                )
                
                # Attention mechanism
                lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
                self.attention = nn.Sequential(
                    nn.Linear(lstm_output_size, lstm_output_size // 2),
                    nn.Tanh(),
                    nn.Linear(lstm_output_size // 2, 1)
                )
                
                # Fully connected layers with batch normalization
                self.fc1 = nn.Linear(lstm_output_size, 64)
                self.bn1 = nn.BatchNorm1d(64)
                self.fc2 = nn.Linear(64, 32)
                self.bn2 = nn.BatchNorm1d(32)
                self.fc3 = nn.Linear(32, 1)
                
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
            
            def forward(self, x):
                # LSTM layers
                lstm_out, _ = self.lstm(x)
                
                # Attention mechanism
                attention_weights = self.attention(lstm_out)
                attention_weights = torch.softmax(attention_weights, dim=1)
                
                # Weighted sum
                attended_out = torch.sum(attention_weights * lstm_out, dim=1)
                
                # Fully connected layers
                out = self.fc1(attended_out)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.dropout(out)
                
                out = self.fc2(out)
                out = self.bn2(out)
                out = self.relu(out)
                out = self.dropout(out)
                
                out = self.fc3(out)
                
                return out.squeeze()
        
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else "cpu")
        
        # Create model based on architecture (optimized original LSTM)
        # Use exact values from model_info.json (Optuna optimized)
        hidden_size1 = lstm_info.get('hidden_size1', 64)
        hidden_size2 = lstm_info.get('hidden_size2', 32)
        dropout = lstm_info.get('dropout', 0.2)
        
        # Debug: print what we're using
        print(f"DEBUG: Creating LSTM model with: hidden_size1={hidden_size1}, hidden_size2={hidden_size2}, dropout={dropout}")
        
        lstm_model = LSTMModel(
            input_size=lstm_info['input_size'],
            hidden_size1=hidden_size1,
            hidden_size2=hidden_size2,
            dropout=dropout
        ).to(device)
        
        # Load state dict with strict=False to see what's wrong if there's a mismatch
        try:
            state_dict = torch.load(model_path, map_location=device)
            lstm_model.load_state_dict(state_dict, strict=True)
            print("DEBUG: Model loaded successfully!")
        except RuntimeError as e:
            # If there's a size mismatch, show what we expected vs what we got
            print(f"DEBUG: Model loading failed: {e}")
            # Try to infer the correct sizes from the state dict
            if 'lstm1.weight_ih_l0' in state_dict:
                actual_hidden1 = state_dict['lstm1.weight_ih_l0'].shape[0] // 4  # LSTM has 4 gates
                print(f"DEBUG: State dict has hidden_size1={actual_hidden1} (from weight shape {state_dict['lstm1.weight_ih_l0'].shape})")
            if 'lstm2.weight_ih_l0' in state_dict:
                actual_hidden2 = state_dict['lstm2.weight_ih_l0'].shape[0] // 4
                print(f"DEBUG: State dict has hidden_size2={actual_hidden2} (from weight shape {state_dict['lstm2.weight_ih_l0'].shape})")
            raise
        lstm_model.eval()
        models['lstm'] = lstm_model
        models['lstm_info'] = lstm_info
        models['lstm_scaler'] = joblib.load(scaler_path)
        models['lstm_device'] = device
        models['lstm_use_improved'] = False
        models['lstm_use_reduced'] = False
        models['lstm_optimized'] = True  # Mark as optimized with Optuna
        
        # No target scaler for original LSTM (not using target normalization)
        models['lstm_target_scaler'] = None
        
        models['lstm_loaded'] = True
    except Exception as e:
        models['lstm_loaded'] = False
        models['lstm_error'] = str(e)
        import traceback
        models['lstm_error_traceback'] = traceback.format_exc()
        print(f"LSTM loading error: {e}")
        print(traceback.format_exc())
    
    # Load Transformer model info and create model
    try:
        with open(models_dir / "transformer_model_info.json", 'r') as f:
            trans_info = json.load(f)
        
        import math
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, max_len=5000):
                super(PositionalEncoding, self).__init__()
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer('pe', pe)
            
            def forward(self, x):
                return x + self.pe[:x.size(0), :]
        
        class TransformerModel(nn.Module):
            def __init__(self, input_size, d_model=128, nhead=8, num_layers=3, dim_feedforward=256, dropout=0.2):
                super(TransformerModel, self).__init__()
                self.input_projection = nn.Linear(input_size, d_model)
                self.pos_encoder = PositionalEncoding(d_model, max_len=5000)
                encoder_layers = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                    dropout=dropout, batch_first=False
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
                self.fc1 = nn.Linear(d_model, 64)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)
                self.fc2 = nn.Linear(64, 1)
            
            def forward(self, x):
                x = self.input_projection(x)
                x = x.transpose(0, 1)
                x = self.pos_encoder(x)
                x = self.transformer_encoder(x)
                x = x[-1, :, :]
                x = self.fc1(x)
                x = self.relu(x)
                x = self.dropout(x)
                x = self.fc2(x)
                return x.squeeze()
        
        device = torch.device("mps" if torch.backends.mps.is_available() else 
                            "cuda" if torch.cuda.is_available() else "cpu")
        trans_model = TransformerModel(
            input_size=trans_info['input_size'],
            d_model=trans_info['d_model'],
            nhead=trans_info['nhead'],
            num_layers=trans_info['num_layers'],
            dim_feedforward=trans_info['dim_feedforward'],
            dropout=trans_info['dropout']
        ).to(device)
        trans_model.load_state_dict(torch.load(models_dir / "transformer_point_model.pth", map_location=device))
        trans_model.eval()
        models['transformer'] = trans_model
        models['transformer_info'] = trans_info
        models['transformer_scaler'] = joblib.load(models_dir / "transformer_scaler.pkl")
        models['transformer_device'] = device
        models['transformer_loaded'] = True
    except Exception as e:
        models['transformer_loaded'] = False
        models['transformer_error'] = str(e)
    
    return models

@st.cache_data
def load_test_data():
    """Load test dataset for quick predictions."""
    processed_dir = project_root / "data" / "processed"
    parquet_file = processed_dir / "rul_features_with_emd.parquet"
    
    if not parquet_file.exists():
        st.error(f"❌ Data file not found: {parquet_file}")
        st.info("""
        **Data file missing!** 
        
        The processed data file (`rul_features_with_emd.parquet`) is not available. 
        This file is excluded from git due to size limitations.
        
        **To fix this:**
        1. Run the data processing notebooks to generate the file
        2. Or upload the file to your Streamlit Cloud deployment
        3. Or use manual feature input mode (coming soon)
        """)
        return None
    
    try:
        df = pd.read_parquet(parquet_file)
        df_clean = df[df['RUL'].notna()].copy()
        test_df = df_clean[df_clean['split'] == 'test'].copy()
        return test_df
    except Exception as e:
        st.error(f"❌ Error loading data file: {str(e)}")
        return None

def clamp_rul_prediction(prediction, model_name="Model", show_warning=True, suppress_warnings=False):
    """
    Clamp RUL prediction to non-negative values.
    RUL (Remaining Useful Life) cannot be negative in practice.
    
    Args:
        prediction: Predicted RUL value (can be float, numpy array, list, etc.)
        model_name: Name of the model for warning messages
        show_warning: Whether to show warning messages (default: True)
        suppress_warnings: If True, only show warnings for extreme cases (default: False)
    
    Returns:
        Clamped prediction (>= 0) - same type as input (float or numpy array)
    """
    # Handle None or invalid inputs
    if prediction is None:
        return 0.0
    
    # Convert to numpy array for consistent handling
    if isinstance(prediction, (list, tuple)):
        pred_array = np.array(prediction, dtype=np.float32)
    elif isinstance(prediction, np.ndarray):
        pred_array = prediction.astype(np.float32)
    else:
        # Scalar value
        try:
            pred_float = float(prediction)
            if np.isnan(pred_float) or np.isinf(pred_float):
                if show_warning and not suppress_warnings:
                    st.warning(f"⚠️ {model_name}: Invalid prediction (NaN/Inf), setting to 0.")
                return 0.0
            if pred_float < 0:
                # Only show warning if not suppressing, or if prediction is very negative (likely error)
                if show_warning and (not suppress_warnings or pred_float < -50):
                    st.warning(f"⚠️ {model_name}: Predicted RUL was negative ({pred_float:.1f} cycles, clamped to 0). "
                              f"This may indicate the battery has reached EOL or the model needs retraining.")
                return 0.0
            return pred_float
        except (ValueError, TypeError):
            if show_warning and not suppress_warnings:
                st.warning(f"⚠️ {model_name}: Could not convert prediction to number, setting to 0.")
            return 0.0
    
    # Handle arrays
    if pred_array.size == 0:
        return np.array([0.0])
    
    # Check for invalid values
    invalid_mask = np.isnan(pred_array) | np.isinf(pred_array)
    if np.any(invalid_mask):
        if show_warning and not suppress_warnings:
            st.warning(f"⚠️ {model_name}: {np.sum(invalid_mask)} prediction(s) were invalid (NaN/Inf), setting to 0.")
        pred_array[invalid_mask] = 0.0
    
    # Clamp negative values
    n_negative = np.sum(pred_array < 0)
    if n_negative > 0:
        # Only warn if many predictions are negative or if suppressing warnings is off
        if show_warning and (not suppress_warnings or n_negative > len(pred_array) * 0.5):
            st.warning(f"⚠️ {model_name}: {n_negative}/{len(pred_array)} prediction(s) were negative (clamped to 0). "
                      f"This may indicate the battery has reached EOL or the model needs retraining.")
        pred_array = np.maximum(pred_array, 0.0)
    
    # Return appropriate type
    if pred_array.size == 1:
        return float(pred_array[0])
    return pred_array

# Load models
with st.spinner("Loading models..."):
    models = load_models()
    test_df = load_test_data()

# Check if test data is available
if test_df is None or test_df.empty:
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Configuration")

# Model selection
st.sidebar.subheader("Select Model")
model_choice = st.sidebar.radio(
    "Choose prediction model:",
    ["Random Forest", "LSTM (with Uncertainty)", "Transformer", "Compare All"],
    index=0
)

# Main content area - Use test data only
# Note: CSV upload removed - it requires processed features (EMD, statistical features)
# which are complex to generate from raw cycle data in the dashboard
st.subheader("📊 Select Test Sample")

# Battery selection
battery_ids = sorted(test_df['battery_id'].unique())
selected_battery = st.selectbox("Select Battery ID:", battery_ids)

# Cycle selection
battery_data = test_df[test_df['battery_id'] == selected_battery].sort_values('cycle_index')
cycles = battery_data['cycle_index'].tolist()
selected_cycle = st.selectbox("Select Cycle:", cycles)

# Get selected row
selected_row = battery_data[battery_data['cycle_index'] == selected_cycle].iloc[0]
actual_rul = selected_row.get('RUL', None)

# Display selected data info
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Battery ID", selected_battery)
with col2:
    st.metric("Cycle", int(selected_cycle))
with col3:
    if actual_rul is not None:
        st.metric("Actual RUL", f"{actual_rul:.1f} cycles")
        # Show EOL status
        if actual_rul < 0:
            st.warning("⚠️ Battery has passed EOL (End of Life)")
    else:
        st.metric("Actual RUL", "N/A")
with col4:
    soh = selected_row.get('SOH', None)
    if soh is not None:
        st.metric("SOH", f"{soh:.3f}")
        if soh < 0.8:
            st.info("🔋 Battery below 80% capacity (EOL threshold)")
    else:
        st.metric("SOH", "N/A")

# Prepare features
exclude_cols = [
    'battery_id', 'filename', 'type', 'start_time', 'test_id', 'uid',
    'split', 'cycle_index', 'EOL_cycle', 'RUL', 'SOH', 'Capacity', 
    'Re', 'Rct', 'ambient_temperature'
]
feature_cols = [c for c in test_df.columns if c not in exclude_cols]

# CRITICAL: Ensure feature order matches model's expected order
# Optimized Random Forest uses full feature set
if models['rf_loaded']:
    # Use model's feature order (optimized with GridSearchCV)
    model_feature_order = list(models['rf'].feature_names_in_)
    X_selected = selected_row[model_feature_order].fillna(0).values.reshape(1, -1)
else:
    X_selected = selected_row[feature_cols].fillna(0).values.reshape(1, -1)
has_actual_rul = actual_rul is not None

# Prediction section
st.divider()
st.subheader("🔮 Predictions")

# Info box about RUL predictions and accuracy
with st.expander("ℹ️ About RUL Predictions & Model Performance"):
    st.markdown("""
    **Remaining Useful Life (RUL)** represents the number of cycles a battery has left before reaching End of Life (EOL).
    
    **Model Performance (Optimized with Hyperparameter Tuning):**
    - **LSTM (Optuna Bayesian Optimization)**: Test MAE **14.72 cycles** ✅, Test RMSE 19.77 cycles, Test R² = 0.206
    - **LSTM with MC Dropout**: Test R² = **0.426** ✅ (Best R² with uncertainty)
    - **Random Forest (GridSearchCV)**: Test MAE **18.82 cycles** ✅, Test RMSE 23.61 cycles, Test R² = **0.244** ✅ (Best R²)
    
    **✅ Current Models in Use**: Optimized with Hyperparameter Tuning
    - **LSTM**: Optimized with Optuna (Bayesian Optimization) - Best MAE
    - **Random Forest**: Optimized with GridSearchCV - Best R²
    - **MC Dropout**: Provides uncertainty quantification with improved R² = 0.426
    
    **Why predictions may be inaccurate:**
    1. **Model limitations**: All models show significant test error (16-27 cycles MAE)
    2. **Overfitting**: Models perform well on training but poorly on test data
       - **LSTM**: Train R² = 0.756, Test R² = 0.159 (only explains 15.9% of variance!)
       - **Random Forest**: Train R² = 0.99, Test R² = -0.06 (also poor generalization)
    3. **LSTM Systematic Bias**: LSTM systematically **over-predicts** RUL by ~12 cycles on average
       - Mean error: +11.9 cycles (73% of predictions are over-predictions)
       - This explains why LSTM predictions are often much higher than actual
       - Example: For actual RUL = 11, LSTM often predicts ~23-39 cycles
    4. **Data distribution shift**: Test batteries may have different degradation patterns
       - Test set has many batteries past EOL (negative RUL values)
       - Models trained on different distribution struggle to generalize
    5. **Early cycles (CRITICAL)**: 
       - **LSTM/Transformer were ONLY trained to predict cycles >= 20** (they need 20 cycles of history)
       - For cycles < 20, predictions are **highly unreliable** because:
         * The model never saw these cycles during training
         * Input sequences must be padded with repeated early cycles (not realistic)
       - **Recommendation**: Use **Random Forest** for early cycles (< 20) - it works better for single-cycle predictions
    6. **Sequence alignment**: LSTM/Transformer predict RUL at cycle N using cycles [N-20 to N-1]
    
    **Why Random Forest may be closer for individual predictions:**
    - Random Forest doesn't have the systematic over-prediction bias of LSTM
    - While it also has poor overall test performance, it can be more accurate for specific cases
    - It doesn't require sequences, so it's more flexible for different cycle ranges
    
    **Negative RUL Handling:**
    - **RUL cannot be negative** in practice - if a battery has reached EOL, RUL = 0
    - Negative predictions are automatically clamped to 0
    - This may indicate the battery has reached EOL or the model needs retraining
    
    **⚠️ Uncertainty Calibration Issues:**
    - **Monte Carlo Dropout intervals are NOT well-calibrated**
    - 90% prediction intervals only achieve ~35% actual coverage (target: 90%)
    - 50% prediction intervals only achieve ~20% actual coverage (target: 50%)
    - Intervals are systematically **too narrow** - they don't capture true variability
    - This is a **critical limitation** - uncertainty estimates are unreliable
    - **Root cause**: Model overfitting + poor generalization to test data
    
    **Recommendations:**
    - Use predictions as estimates, not exact values
    - **Do NOT rely on uncertainty intervals** - they are poorly calibrated
    - For early cycles (< 20), predictions are less reliable due to padding
    - Models work best for cycles 20+ where full sequences are available
    - Consider predictions as rough estimates with ±20 cycle error range
    """)

if model_choice == "Random Forest" and models['rf_loaded']:
    # Random Forest prediction
    pred_rf_raw = models['rf'].predict(X_selected)[0]
    
    # Show raw prediction for debugging
    if st.checkbox("🔍 Show Raw Prediction (Debug)", key="rf_debug"):
        st.info(f"Raw prediction (before clamping): {pred_rf_raw:.2f} cycles")
        if has_actual_rul:
            st.info(f"Actual RUL: {actual_rul:.2f} cycles")
            st.info(f"Error: {abs(pred_rf_raw - actual_rul):.2f} cycles")
    
    # Suppress warnings if battery has passed EOL (negative actual RUL is expected)
    suppress_warnings = has_actual_rul and actual_rul < 0
    pred_rf = clamp_rul_prediction(pred_rf_raw, "Random Forest", suppress_warnings=suppress_warnings)
    
    # Explain why prediction might be 0
    if pred_rf == 0 and pred_rf_raw < 0:
        if has_actual_rul and actual_rul < 0:
            st.info("ℹ️ **Why prediction is 0:** The model predicted negative RUL (battery past EOL), which was clamped to 0. "
                   f"Raw prediction was {pred_rf_raw:.1f} cycles, actual RUL is {actual_rul:.1f} cycles. "
                   "For batteries past EOL, RUL is set to 0 in practice.")
        else:
            st.warning("⚠️ **Why prediction is 0:** The model predicted negative RUL, which was clamped to 0. "
                      f"Raw prediction was {pred_rf_raw:.1f} cycles. This may indicate model error or battery has reached EOL.")
    
    col1, col2 = st.columns(2)
    with col1:
        if has_actual_rul:
            delta_val = pred_rf - actual_rul
            st.metric("Predicted RUL", f"{pred_rf:.1f} cycles", delta=f"{delta_val:.1f}")
        else:
            st.metric("Predicted RUL", f"{pred_rf:.1f} cycles")
    with col2:
        if has_actual_rul:
            st.metric("Actual RUL", f"{actual_rul:.1f} cycles")
            # Show if battery has passed EOL
            if actual_rul < 0:
                st.warning("⚠️ This battery has passed EOL (negative RUL). Predictions may be less reliable.")
        else:
            st.metric("Actual RUL", "N/A (uploaded data)")
    
    # Feature importance
    if st.checkbox("Show Feature Importance"):
        feature_importance = pd.DataFrame({
            'feature': models['rf'].feature_names_in_,
            'importance': models['rf'].feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_importance)), feature_importance['importance'], color='#2ecc71')
        ax.set_yticks(range(len(feature_importance)))
        ax.set_yticklabels(feature_importance['feature'])
        ax.set_xlabel('Importance')
        ax.set_title('Top 10 Feature Importance')
        ax.invert_yaxis()
        st.pyplot(fig)

elif model_choice == "LSTM (with Uncertainty)":
    if not models.get('lstm_loaded', False):
        st.error(f"❌ LSTM model not loaded. Error: {models.get('lstm_error', 'Unknown error')}")
        if 'lstm_error_traceback' in models:
            with st.expander("🔍 Show detailed error"):
                st.code(models['lstm_error_traceback'])
        st.info("💡 **Troubleshooting:**\n"
                "- Check that `lstm_pytorch_model_info.json` exists\n"
                "- Check that `lstm_pytorch_point_model.pth` exists\n"
                "- Check that `lstm_pytorch_scaler.pkl` exists\n"
                "- Verify all files are in `results/models/` directory")
    elif models['lstm_loaded']:
        # LSTM prediction with MC Dropout
        sequence_length = models['lstm_info']['sequence_length']
        
        # IMPORTANT: Model was trained to predict RUL at cycle i+20 given cycles [i to i+19]
        # So to predict RUL at cycle N, we need cycles [N-20 to N-1] (20 cycles before N)
        # The model was ONLY trained on sequences where target cycle >= 20
        # For cycles < 20, predictions are unreliable because:
        # 1. The model never saw these cycles during training
        # 2. We must pad sequences with repeated early cycles, which is not realistic
        
        # Warn user if predicting for early cycles
        if selected_cycle < sequence_length:
            st.warning(f"⚠️ **Early Cycle Warning:** You are predicting RUL for cycle {selected_cycle}, but the LSTM model was only trained to predict cycles >= {sequence_length}. "
                      f"Predictions for cycles < {sequence_length} are **unreliable** because:\n"
                      f"- The model never saw these cycles during training\n"
                      f"- Input sequences must be padded with repeated early cycles (not realistic)\n\n"
                      f"**Recommendation:** Use **Random Forest** for early cycles (< {sequence_length}) as it doesn't require sequences and works better for early cycle predictions.")
        
        battery_sequences = test_df[test_df['battery_id'] == selected_battery].sort_values('cycle_index')
        
        # Get cycles before selected cycle (we predict RUL AT selected cycle, so we need cycles BEFORE it)
        start_cycle = max(1, selected_cycle - sequence_length + 1)
        cycles_for_prediction = battery_sequences[
            (battery_sequences['cycle_index'] >= start_cycle) & 
            (battery_sequences['cycle_index'] < selected_cycle)
        ]
        
        if len(cycles_for_prediction) > 0 or selected_cycle == 1:
            # Build sequence of exactly sequence_length cycles
            if len(cycles_for_prediction) >= sequence_length:
                # Take last sequence_length cycles before selected cycle
                seq_cycles = cycles_for_prediction.tail(sequence_length)
            else:
                # Need to pad - use first available cycle to pad from the beginning
                if len(cycles_for_prediction) > 0:
                    first_cycle = cycles_for_prediction.iloc[0:1]
                    seq_cycles = pd.concat([first_cycle] * (sequence_length - len(cycles_for_prediction)) + [cycles_for_prediction])
                else:
                    # No cycles before selected cycle - use selected cycle itself for padding
                    selected_cycle_data = battery_sequences[battery_sequences['cycle_index'] == selected_cycle]
                    if len(selected_cycle_data) > 0:
                        seq_cycles = pd.concat([selected_cycle_data] * sequence_length)
                    else:
                        st.error("Cannot create sequence - no data available")
                        st.stop()
            
            # Prepare features (using full feature set)
            seq_scaled = prepare_lstm_features(seq_cycles, models, feature_cols)
            # Ensure correct shape: (1, sequence_length, n_features)
            if seq_scaled.shape[0] == 1:
                seq_scaled = seq_scaled[0:1]  # Keep batch dimension
            else:
                seq_scaled = seq_scaled.reshape(1, sequence_length, -1)
            
            # Ensure model is in eval mode before point prediction (BatchNorm needs this for batch size 1)
            models['lstm'].eval()

            # Point prediction
            with torch.no_grad():
                X_tensor = torch.FloatTensor(seq_scaled).to(models['lstm_device'])
                pred_lstm_raw_scaled = models['lstm'](X_tensor).cpu().numpy()
                # Handle scalar vs array output
                if pred_lstm_raw_scaled.ndim == 0:
                    pred_lstm_raw_scaled = float(pred_lstm_raw_scaled)
                else:
                    pred_lstm_raw_scaled = pred_lstm_raw_scaled[0] if len(pred_lstm_raw_scaled) > 0 else float(pred_lstm_raw_scaled)
            
            # Optimized original LSTM doesn't use target normalization
            pred_lstm_raw = pred_lstm_raw_scaled
            
            # Show raw prediction for debugging
            if st.checkbox("🔍 Show Raw Prediction (Debug)", key="lstm_debug"):
                st.info(f"Raw prediction (before clamping): {pred_lstm_raw:.2f} cycles")
                if models.get('lstm_target_scaler') is not None:
                    st.info(f"Scaled prediction: {pred_lstm_raw_scaled:.2f} (denormalized)")
                st.info(f"Sequence used: cycles {start_cycle} to {selected_cycle-1} (padded if needed)")
                if models.get('lstm_optimized', False):
                    st.info("✅ Using OPTIMIZED LSTM model (Optuna Bayesian Optimization: MAE 14.72, R² 0.206)")
            
            # Clamp negative predictions (point prediction)
            suppress_warnings = has_actual_rul and actual_rul < 0
            pred_lstm = clamp_rul_prediction(pred_lstm_raw, "LSTM", suppress_warnings=suppress_warnings)
            
            # MC Dropout for uncertainty
            n_mc_samples = st.sidebar.slider("MC Dropout Samples", 10, 200, 100, 10)
            
            enable_dropout_during_inference(models['lstm'])  # Enable dropout without BatchNorm issues
            mc_predictions_raw_scaled = []
            with torch.no_grad():
                for _ in range(n_mc_samples):
                    pred = models['lstm'](X_tensor).cpu().numpy()
                    # Handle scalar vs array output
                    if pred.ndim == 0:
                        pred_val = float(pred)
                    else:
                        pred_val = pred[0] if len(pred) > 0 else float(pred)
                    mc_predictions_raw_scaled.append(pred_val)
            
            models['lstm'].eval()  # Restore eval mode
            
            # Optimized original LSTM doesn't use target normalization
            mc_predictions_raw = mc_predictions_raw_scaled
            target_scale = 1.0
            
            # Clamp all MC predictions to non-negative
            suppress_warnings = has_actual_rul and actual_rul < 0
            mc_predictions = clamp_rul_prediction(mc_predictions_raw, "LSTM (MC Dropout)", suppress_warnings=suppress_warnings)
            
            # Calculate statistics from clamped predictions
            pred_mean = float(np.mean(mc_predictions))
            pred_std = float(np.std(mc_predictions))  # Already on original scale after denormalization
            pred_q05 = float(np.percentile(mc_predictions, 5))
            pred_q95 = float(np.percentile(mc_predictions, 95))
            
            # Display results
            col1, col2, col3 = st.columns(3)
            with col1:
                if has_actual_rul:
                    delta_val = pred_mean - actual_rul
                    st.metric("Predicted RUL", f"{pred_mean:.1f} cycles", delta=f"{delta_val:.1f}")
                else:
                    st.metric("Predicted RUL", f"{pred_mean:.1f} cycles")
            with col2:
                st.metric("Uncertainty (std)", f"{pred_std:.2f} cycles")
            with col3:
                st.metric("90% Interval", f"[{pred_q05:.1f}, {pred_q95:.1f}]")
            
            # Calibration warning
            if has_actual_rul:
                # Check if actual RUL is within the 90% interval
                in_interval_90 = pred_q05 <= actual_rul <= pred_q95
                if not in_interval_90:
                    st.warning(
                        f"⚠️ **Uncertainty Calibration Issue**: The actual RUL ({actual_rul:.1f} cycles) is **outside** the 90% confidence interval "
                        f"[{pred_q05:.1f}, {pred_q95:.1f}]. This indicates the uncertainty intervals are **not well-calibrated**. "
                        f"Research shows actual coverage is ~35% (target: 90%). Use intervals with caution."
                    )
            
            # Updated calibration info
            st.success(
                "📊 **MC Dropout Performance**: With optimized model, MC Dropout achieves R² = 0.426 (improved from baseline R² = 0.206). "
                "Uncertainty intervals provide better calibration with the optimized hyperparameters."
            )
            
            # Visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.axvspan(pred_q05, pred_q95, alpha=0.2, color='blue', label='90% Confidence Interval')
            ax.axvline(pred_mean, color='blue', linewidth=2, label='Mean Prediction')
            if has_actual_rul:
                ax.axvline(actual_rul, color='red', linewidth=2, linestyle='--', label='Actual RUL')
            ax.set_xlabel('RUL (cycles)')
            ax.set_ylabel('Density')
            ax.set_title('LSTM Prediction with Uncertainty')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
        else:
            st.warning(f"No cycles available up to cycle {selected_cycle} for battery {selected_battery}.")

elif model_choice == "Transformer" and models['transformer_loaded']:
    # Transformer prediction
    sequence_length = models['transformer_info']['sequence_length']
    
    # Same logic as LSTM: predict RUL at cycle N using cycles [N-20 to N-1]
    # The model was ONLY trained on sequences where target cycle >= 20
    # For cycles < 20, predictions are unreliable because:
    # 1. The model never saw these cycles during training
    # 2. We must pad sequences with repeated early cycles, which is not realistic
    
    # Warn user if predicting for early cycles
    if selected_cycle < sequence_length:
        st.warning(f"⚠️ **Early Cycle Warning:** You are predicting RUL for cycle {selected_cycle}, but the Transformer model was only trained to predict cycles >= {sequence_length}. "
                  f"Predictions for cycles < {sequence_length} are **unreliable** because:\n"
                  f"- The model never saw these cycles during training\n"
                  f"- Input sequences must be padded with repeated early cycles (not realistic)\n\n"
                  f"**Recommendation:** Use **Random Forest** for early cycles (< {sequence_length}) as it doesn't require sequences and works better for early cycle predictions.")
    
    battery_sequences = test_df[test_df['battery_id'] == selected_battery].sort_values('cycle_index')
    
    start_cycle = max(1, selected_cycle - sequence_length + 1)
    cycles_for_prediction = battery_sequences[
        (battery_sequences['cycle_index'] >= start_cycle) & 
        (battery_sequences['cycle_index'] < selected_cycle)
    ]
    
    if len(cycles_for_prediction) > 0 or selected_cycle == 1:
        if len(cycles_for_prediction) >= sequence_length:
            seq_data = cycles_for_prediction.tail(sequence_length)[feature_cols].fillna(0).values.astype(np.float32)
        else:
            if len(cycles_for_prediction) > 0:
                first_available = cycles_for_prediction.iloc[0][feature_cols].fillna(0).values.astype(np.float32)
                available_data = cycles_for_prediction[feature_cols].fillna(0).values.astype(np.float32)
            else:
                selected_cycle_data = battery_sequences[battery_sequences['cycle_index'] == selected_cycle]
                if len(selected_cycle_data) > 0:
                    first_available = selected_cycle_data.iloc[0][feature_cols].fillna(0).values.astype(np.float32)
                    available_data = selected_cycle_data[feature_cols].fillna(0).values.astype(np.float32)
                else:
                    st.error("Cannot create sequence - no data available")
                    st.stop()
            
            n_pad = sequence_length - len(available_data)
            padded_data = np.vstack([np.tile(first_available, (n_pad, 1)), available_data])
            seq_data = padded_data
        
        # Normalize
        seq_reshaped = seq_data.reshape(-1, len(feature_cols))
        seq_scaled = models['transformer_scaler'].transform(seq_reshaped)
        seq_scaled = seq_scaled.reshape(1, sequence_length, len(feature_cols))
        
        # Prediction
        with torch.no_grad():
            X_tensor = torch.FloatTensor(seq_scaled).to(models['transformer_device'])
            pred_trans_raw = models['transformer'](X_tensor).cpu().numpy()
            # Handle scalar vs array output
            if pred_trans_raw.ndim == 0:
                pred_trans_raw = float(pred_trans_raw)
            else:
                pred_trans_raw = pred_trans_raw[0] if len(pred_trans_raw) > 0 else float(pred_trans_raw)
        
        # Show raw prediction for debugging
        if st.checkbox("🔍 Show Raw Prediction (Debug)", key="trans_debug"):
            st.info(f"Raw prediction (before clamping): {pred_trans_raw:.2f} cycles")
            st.info(f"Sequence used: cycles {start_cycle} to {selected_cycle-1} (padded if needed)")
        
        # Clamp negative predictions
        suppress_warnings = has_actual_rul and actual_rul < 0
        pred_trans = clamp_rul_prediction(pred_trans_raw, "Transformer", suppress_warnings=suppress_warnings)
        
        col1, col2 = st.columns(2)
        with col1:
            if has_actual_rul:
                delta_val = pred_trans - actual_rul
                st.metric("Predicted RUL", f"{pred_trans:.1f} cycles", delta=f"{delta_val:.1f}")
            else:
                st.metric("Predicted RUL", f"{pred_trans:.1f} cycles")
        with col2:
            if has_actual_rul:
                st.metric("Actual RUL", f"{actual_rul:.1f} cycles")
            else:
                st.metric("Actual RUL", "N/A (uploaded data)")
    else:
        st.warning(f"No cycles available up to cycle {selected_cycle} for battery {selected_battery}.")

elif model_choice == "Compare All":
    # Compare all models
    st.subheader("📊 Model Comparison")
    
    # Warn user if predicting for early cycles (LSTM/Transformer limitation)
    sequence_length = models['lstm_info']['sequence_length'] if models['lstm_loaded'] else (models['transformer_info']['sequence_length'] if models['transformer_loaded'] else 20)
    if selected_cycle < sequence_length:
        st.warning(f"⚠️ **Early Cycle Warning:** You are comparing models for cycle {selected_cycle}, but LSTM/Transformer were only trained to predict cycles >= {sequence_length}. "
                  f"Predictions for cycles < {sequence_length} are **unreliable** for LSTM/Transformer because:\n"
                  f"- The models never saw these cycles during training\n"
                  f"- Input sequences must be padded with repeated early cycles (not realistic)\n\n"
                  f"**Recommendation:** For early cycles (< {sequence_length}), **Random Forest** is more reliable as it doesn't require sequences.")
    
    predictions = {}
    
    # Random Forest
    if models['rf_loaded']:
        pred_rf_raw = models['rf'].predict(X_selected)[0]
        suppress_warnings = has_actual_rul and actual_rul < 0
        predictions['Random Forest'] = clamp_rul_prediction(pred_rf_raw, "Random Forest", suppress_warnings=suppress_warnings)
    
    # LSTM and Transformer need sequences - using test data
    # LSTM
        if models['lstm_loaded']:
            sequence_length = models['lstm_info']['sequence_length']
            battery_sequences = test_df[test_df['battery_id'] == selected_battery].sort_values('cycle_index')
            
            start_cycle = max(1, selected_cycle - sequence_length + 1)
            cycles_for_prediction = battery_sequences[
                (battery_sequences['cycle_index'] >= start_cycle) & 
                (battery_sequences['cycle_index'] < selected_cycle)
            ]
            
            if len(cycles_for_prediction) > 0 or selected_cycle == 1:
                if len(cycles_for_prediction) >= sequence_length:
                    seq_cycles = cycles_for_prediction.tail(sequence_length)
                else:
                    if len(cycles_for_prediction) > 0:
                        first_cycle = cycles_for_prediction.iloc[0:1]
                        seq_cycles = pd.concat([first_cycle] * (sequence_length - len(cycles_for_prediction)) + [cycles_for_prediction])
                    else:
                        selected_cycle_data = battery_sequences[battery_sequences['cycle_index'] == selected_cycle]
                        seq_cycles = pd.concat([selected_cycle_data] * sequence_length)
                
                # Prepare features (using full feature set)
                seq_scaled = prepare_lstm_features(seq_cycles, models, feature_cols)
                seq_scaled = seq_scaled.reshape(1, sequence_length, -1)
                # Ensure model is in eval mode
                models['lstm'].eval()
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(seq_scaled).to(models['lstm_device'])
                    pred_lstm_scaled = models['lstm'](X_tensor).cpu().numpy()
                    # Handle scalar vs array output
                    if pred_lstm_scaled.ndim == 0:
                        pred_lstm_scaled_val = float(pred_lstm_scaled)
                    else:
                        pred_lstm_scaled_val = pred_lstm_scaled[0] if len(pred_lstm_scaled) > 0 else float(pred_lstm_scaled)
                    
                    # Optimized original LSTM doesn't use target normalization
                    pred_lstm_val = pred_lstm_scaled_val
                    
                    suppress_warnings = has_actual_rul and actual_rul < 0
                    predictions['LSTM'] = clamp_rul_prediction(pred_lstm_val, "LSTM", suppress_warnings=suppress_warnings)
            else:
                st.warning(f"No cycles available up to cycle {selected_cycle} for LSTM.")
    
    # Transformer
        if models['transformer_loaded']:
            sequence_length = models['transformer_info']['sequence_length']
            battery_sequences = test_df[test_df['battery_id'] == selected_battery].sort_values('cycle_index')
            
            start_cycle = max(1, selected_cycle - sequence_length + 1)
            cycles_for_prediction = battery_sequences[
                (battery_sequences['cycle_index'] >= start_cycle) & 
                (battery_sequences['cycle_index'] < selected_cycle)
            ]
            
            if len(cycles_for_prediction) > 0 or selected_cycle == 1:
                if len(cycles_for_prediction) >= sequence_length:
                    seq_data = cycles_for_prediction.tail(sequence_length)[feature_cols].fillna(0).values.astype(np.float32)
                else:
                    if len(cycles_for_prediction) > 0:
                        first_available = cycles_for_prediction.iloc[0][feature_cols].fillna(0).values.astype(np.float32)
                        available_data = cycles_for_prediction[feature_cols].fillna(0).values.astype(np.float32)
                    else:
                        selected_cycle_data = battery_sequences[battery_sequences['cycle_index'] == selected_cycle]
                        first_available = selected_cycle_data.iloc[0][feature_cols].fillna(0).values.astype(np.float32)
                        available_data = selected_cycle_data[feature_cols].fillna(0).values.astype(np.float32)
                    
                    n_pad = sequence_length - len(available_data)
                    padded_data = np.vstack([np.tile(first_available, (n_pad, 1)), available_data])
                    seq_data = padded_data
                seq_reshaped = seq_data.reshape(-1, len(feature_cols))
                seq_scaled = models['transformer_scaler'].transform(seq_reshaped)
                seq_scaled = seq_scaled.reshape(1, sequence_length, len(feature_cols))
                with torch.no_grad():
                    X_tensor = torch.FloatTensor(seq_scaled).to(models['transformer_device'])
                    pred_trans = models['transformer'](X_tensor).cpu().numpy()
                    # Handle scalar vs array output
                    if pred_trans.ndim == 0:
                        pred_trans_val = float(pred_trans)
                    else:
                        pred_trans_val = pred_trans[0] if len(pred_trans) > 0 else float(pred_trans)
                    suppress_warnings = has_actual_rul and actual_rul < 0
                    predictions['Transformer'] = clamp_rul_prediction(pred_trans_val, "Transformer", suppress_warnings=suppress_warnings)
        else:
            st.warning(f"No cycles available up to cycle {selected_cycle} for Transformer.")
    
    # Display comparison
    if predictions:
        comparison_df = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Predicted RUL': list(predictions.values())
        })
        if has_actual_rul:
            comparison_df['Actual RUL'] = actual_rul
            comparison_df['Error'] = comparison_df['Predicted RUL'] - comparison_df['Actual RUL']
        
        if has_actual_rul:
            st.dataframe(comparison_df.style.format({
                'Predicted RUL': '{:.1f}',
                'Actual RUL': '{:.1f}',
                'Error': '{:.1f}'
            }))
        else:
            st.dataframe(comparison_df.style.format({
                'Predicted RUL': '{:.1f}'
            }))
        
        # Visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        x_pos = np.arange(len(predictions))
        colors = ['#2ecc71', '#3498db', '#e74c3c'][:len(predictions)]
        ax.bar(x_pos, list(predictions.values()), alpha=0.7, color=colors)
        if has_actual_rul:
            ax.axhline(actual_rul, color='red', linestyle='--', linewidth=2, label='Actual RUL')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(list(predictions.keys()))
        ax.set_ylabel('RUL (cycles)')
        ax.set_title('Model Comparison: Predicted RUL')
        if has_actual_rul:
            ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        # Calibration warning if LSTM is included
        if 'LSTM' in predictions:
            st.info(
                "📊 **Note on LSTM Uncertainty**: If using LSTM with uncertainty quantification, "
                "the confidence intervals are **not well-calibrated** (actual coverage ~35% vs target 90%). "
                "See the 'About RUL Predictions' section above for details."
            )

else:
    if model_choice == "LSTM (with Uncertainty)":
        # Error already shown above
        pass
    else:
        st.error("Model not loaded. Please check if model files exist in results/models/")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>Intelligent Battery Monitoring System | RUL Prediction with Uncertainty Quantification</p>
</div>
""", unsafe_allow_html=True)

