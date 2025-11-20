# Feature Reduction Analysis: Full Features vs PCA-Reduced Features

## Summary

I conducted an analysis comparing models trained on full features (175) versus PCA-reduced features (57 components). After thorough evaluation, I decided to use **full features with hyperparameter optimization** as it provides better overall performance.

## Comparison Results

### LSTM Model Comparison

| Metric | Full Features (175) | Reduced (57 PCA) | Winner |
|--------|---------------------|------------------|--------|
| **Test MAE** | **14.72 cycles** ✅ | 15.54 cycles | **Full Features** |
| **Test RMSE** | **19.77 cycles** ✅ | 20.51 cycles | **Full Features** |
| **Test R²** | **0.206** ✅ | 0.145 | **Full Features** |

**Analysis:**
- Full features achieve better MAE (5% improvement)
- Full features achieve better RMSE (4% improvement)
- Full features achieve better R² (42% improvement)
- **Verdict**: **Use Full Features** - Better performance across all metrics

### Random Forest Model Comparison

| Metric | Full Features (175) | Reduced (57 PCA) | Winner |
|--------|---------------------|------------------|--------|
| **Test MAE** | **18.82 cycles** ✅ | 20.08 cycles | **Full Features** |
| **Test RMSE** | **23.61 cycles** ✅ | 24.89 cycles | **Full Features** |
| **Test R²** | **0.244** ✅ | 0.160 | **Full Features** |

**Analysis:**
- Full features achieve better MAE (6% improvement)
- Full features achieve better RMSE (5% improvement)
- Full features achieve better R² (53% improvement)
- **Verdict**: **Use Full Features** - Consistently better performance

## Decision: Use Full Features

After comparing both approaches with hyperparameter optimization:

### **Full Features (175) - Selected Approach** ✅

**Advantages:**
- ✅ Better MAE for both models
- ✅ Better RMSE for both models
- ✅ Better R² for both models
- ✅ More information available to models
- ✅ No information loss from dimensionality reduction

**With Hyperparameter Optimization:**
- Random Forest: GridSearchCV optimized (R²: 0.244)
- LSTM: Optuna optimized (MAE: 14.72, R²: 0.206)
- MC Dropout: R² improved to 0.426

### **PCA-Reduced Features (57) - Not Used**

**Why I didn't use it:**
- ❌ Lower performance across all metrics
- ❌ Information loss from dimensionality reduction
- ❌ Additional complexity (need to maintain PCA transformers)
- ❌ No clear advantage over full features

## Key Insights

1. **Full Features Work Better**: With proper hyperparameter optimization, full features outperform reduced features
2. **Optimization Matters**: GridSearchCV and Optuna found optimal parameters that work well with full feature set
3. **No Need for Reduction**: 175 features is manageable and provides better information to models
4. **Simplicity**: Using full features eliminates need for PCA preprocessing

## Current Implementation

I'm using:
- **Full feature set**: 175 features (16 statistical + 159 EMD)
- **Random Forest**: GridSearchCV optimized with full features
- **LSTM**: Optuna optimized with full features
- **No PCA reduction**: Direct use of all features

This approach provides the best performance and keeps the system simple and maintainable.
