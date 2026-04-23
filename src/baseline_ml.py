import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    r2_score, mean_absolute_percentage_error
)
import warnings, pickle
warnings.filterwarnings('ignore')

out = Path('../data/results/baseline ML')
out.mkdir(exist_ok=True)

# LOAD & PREPARE
df = pd.read_csv('../data/processed/Monthly_Demand_Dataset.csv')

df.replace([np.inf, -np.inf], 0, inplace=True)
df.fillna(0, inplace=True)

le = LabelEncoder()
df['Category_Encoded'] = le.fit_transform(df['Category'])

feature_cols = [
    'Qty_Lag_1', 'Qty_Lag_2', 'Qty_Lag_3',
    'Qty_Rolling_3M_Avg',
    'Avg_Unit_Price', 'Median_Unit_Price',
    'Min_Unit_Price', 'Max_Unit_Price',
    'Avg_Qty_Per_Order',
    'Price_Change_Pct', 'Qty_Growth_Pct',
    'Lifetime_Total_Qty', 'Lifetime_Avg_Price',
    'Months_Since_Launch',
    'Month',
    'Category_Encoded',
]
target = 'Total_Qty_Sold'


# TIME-BASED TRAIN/TEST SPLIT
train_mask = df['Year_Month_Str'] < '2025-07'
test_mask = df['Year_Month_Str'] >= '2025-07'

X_train = df.loc[train_mask, feature_cols]
y_train = df.loc[train_mask, target]
X_test = df.loc[test_mask, feature_cols]
y_test = df.loc[test_mask, target]

print(f"Train: {len(X_train):,} rows ({df.loc[train_mask, 'Year_Month_Str'].min()} to {df.loc[train_mask, 'Year_Month_Str'].max()})")
print(f"Test:  {len(X_test):,} rows ({df.loc[test_mask, 'Year_Month_Str'].min()} to {df.loc[test_mask, 'Year_Month_Str'].max()})")

# TRAIN MULTIPLE MODELS
models = {
    'Ridge Regression': Ridge(alpha=1.0),
    'Random Forest': RandomForestRegressor(
        n_estimators=300, max_depth=12, min_samples_leaf=5,
        random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        min_samples_leaf=5, subsample=0.8, random_state=42
    ),
    'Hist Gradient Boosting': HistGradientBoostingRegressor(
        max_iter=500, max_depth=8, learning_rate=0.05,
        min_samples_leaf=10, random_state=42
    ),
}

results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    y_pred_train = np.maximum(model.predict(X_train), 0)
    y_pred_test = np.maximum(model.predict(X_test), 0)

    predictions[name] = y_pred_test

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    mask_nz = y_test > 0
    mape = mean_absolute_percentage_error(y_test[mask_nz], y_pred_test[mask_nz]) * 100

    results[name] = {
        'MAE': mae, 'RMSE': rmse, 'R²_Test': r2,
        'R²_Train': r2_train, 'MAPE_%': mape
    }
    print(f"  Train R²: {r2_train:.4f} | Test R²: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | MAPE: {mape:.1f}%")

# RESULTS TABLE
results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values('MAE')
results_df.to_csv(out / 'model_comparison.csv')

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(results_df.to_string())

best_model_name = results_df.index[0]
best_model = models[best_model_name]
print(f"\n>>> Best Model: {best_model_name} <<<")

# Save best model
with open(out / 'best_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'features': feature_cols, 'label_encoder': le}, f)

# MODEL COMPARISON
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
metrics_list = ['MAE', 'RMSE', 'R²_Test']
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

for idx, metric in enumerate(metrics_list):
    vals = results_df[metric]
    axes[idx].barh(vals.index, vals.values, color=colors[:len(vals)])
    axes[idx].set_title(metric, fontsize=13, fontweight='bold')
    for i, v in enumerate(vals.values):
        axes[idx].text(v + 0.01 * max(abs(vals)), i, f'{v:.3f}', va='center', fontsize=10)

plt.suptitle('Model Comparison', fontsize=15, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out / 'model_comparison.png', dpi=150)
plt.close()

# FEATURE IMPORTANCE
if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
else:
    importances = np.abs(best_model.coef_)

feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
feat_imp.plot(kind='barh', ax=ax, color='#4C72B0')
ax.set_title(f'Feature Importance — {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance')
for i, v in enumerate(feat_imp.values):
    ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(out / 'feature_importance.png', dpi=150)
plt.close()

# ACTUAL VS PREDICTED
best_preds = predictions[best_model_name]

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, best_preds, alpha=0.3, s=15, color='#4C72B0')
max_val = max(y_test.max(), best_preds.max())
ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlabel('Actual Qty Sold', fontsize=12)
ax.set_ylabel('Predicted Qty Sold', fontsize=12)
ax.set_title(f'Actual vs Predicted — {best_model_name}\n(R² = {results[best_model_name]["R²_Test"]:.4f})',
             fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
lim = min(200, max_val)
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
plt.tight_layout()
plt.savefig(out / 'actual_vs_predicted.png', dpi=150)
plt.close()

# RESIDUAL ANALYSIS
residuals = y_test.values - best_preds

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].scatter(best_preds, residuals, alpha=0.3, s=15, color='#4C72B0')
axes[0].axhline(0, color='red', linewidth=2, linestyle='--')
axes[0].set_xlabel('Predicted Qty')
axes[0].set_ylabel('Residual (Actual - Predicted)')
axes[0].set_title('Residuals vs Predicted')
axes[0].set_xlim(0, 150)
axes[0].set_ylim(-100, 100)

axes[1].hist(residuals, bins=60, color='#55A868', edgecolor='white')
axes[1].axvline(0, color='red', linewidth=2, linestyle='--')
axes[1].set_xlabel('Residual')
axes[1].set_ylabel('Frequency')
axes[1].set_title(f'Residual Distribution (Mean: {residuals.mean():.2f})')
axes[1].set_xlim(-100, 100)

plt.suptitle(f'Residual Analysis — {best_model_name}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out / 'residual_analysis.png', dpi=150)
plt.close()

# PERFORMANCE BY CATEGORY
test_df = df.loc[test_mask].copy()
test_df['Predicted'] = best_preds

cat_perf = test_df.groupby('Category').apply(
    lambda g: pd.Series({
        'MAE': mean_absolute_error(g[target], g['Predicted']),
        'R²': r2_score(g[target], g['Predicted']) if len(g) > 1 else 0,
        'Count': len(g)
    })
).sort_values('MAE')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

cat_perf['MAE'].plot(kind='barh', ax=axes[0], color='#4C72B0')
axes[0].set_title('MAE by Category', fontsize=13, fontweight='bold')
axes[0].set_xlabel('MAE')
for i, v in enumerate(cat_perf['MAE']):
    axes[0].text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=10)

cat_perf['R²'].plot(kind='barh', ax=axes[1], color='#55A868')
axes[1].set_title('R² by Category', fontsize=13, fontweight='bold')
axes[1].set_xlabel('R²')
for i, v in enumerate(cat_perf['R²']):
    axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=10)

plt.suptitle(f'Performance by Category — {best_model_name}', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out / 'performance_by_category.png', dpi=150)
plt.close()

cat_perf.to_csv(out / 'performance_by_category.csv')

# SAVE TEST PREDICTIONS
test_output = test_df[['Item ID', 'Item_Description', 'Category', 'Year_Month_Str',
                        target, 'Predicted']].copy()
test_output['Predicted'] = test_output['Predicted'].round(1)
test_output['Error'] = (test_output[target] - test_output['Predicted']).round(1)
test_output.to_csv(out / 'test_predictions.csv', index=False)

print("\n" + "="*70)
print("TRAINING COMPLETE — FINAL SUMMARY")
print("="*70)
print(f"\nBest Model: {best_model_name}")
print(f"  Test R²:   {results[best_model_name]['R²_Test']:.4f}")
print(f"  Test MAE:  {results[best_model_name]['MAE']:.2f} units")
print(f"  Test RMSE: {results[best_model_name]['RMSE']:.2f} units")
print(f"  MAPE:      {results[best_model_name]['MAPE_%']:.1f}%")
print(f"  Train R²:  {results[best_model_name]['R²_Train']:.4f}")
print(f"  Overfit gap: {results[best_model_name]['R²_Train'] - results[best_model_name]['R²_Test']:.4f}")
print(f"\nTop 5 Features:")
for feat, imp in feat_imp.sort_values(ascending=False).head(5).items():
    print(f"  {feat}: {imp:.4f}")
print(f"\nPerformance by Category:")
print(cat_perf.to_string())
