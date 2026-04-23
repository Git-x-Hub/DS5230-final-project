"""
DeepAR Demand Forecasting — GluonTS (PyTorch Backend)

Key design decisions:
  - Only trains on products with 18+ months of history (reduces noise)
  - Uses NegativeBinomial distribution (better for count/demand data)
  - Builds continuous time series per product (no artificial zero-padding)
  - Reports metrics on ALL products (DeepAR products + fallback for sparse ones)

Requirements: pip install "gluonts[torch]" pandas numpy matplotlib scikit-learn lightning
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import warnings, json
warnings.filterwarnings('ignore')

import torch
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions
from gluonts.time_feature import month_of_year
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# CONFIGURATION
PREDICTION_LENGTH = 7          # Forecast 7 months (Jul 2025 – Jan 2026)
CONTEXT_LENGTH = 24            # Look back 24 months
TRAIN_CUTOFF = '2025-07'
MIN_HISTORY_MONTHS = 18        # Only use products with enough history for DeepAR
EPOCHS = 150
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_LAYERS = 2
HIDDEN_SIZE = 64
DROPOUT = 0.1
NUM_SAMPLES = 200
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

out = Path('../data/results/advanced DeepAr')
out.mkdir(exist_ok=True)

print("DeepAR DEMAND FORECASTING")
print(f"Device: {DEVICE}")
if DEVICE == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Prediction length: {PREDICTION_LENGTH} months")
print(f"Context length: {CONTEXT_LENGTH} months")
print(f"Min history: {MIN_HISTORY_MONTHS} months")
print(f"Epochs: {EPOCHS}")

# LOAD AND FILTER DATA
df = pd.read_csv('../data/processed/Monthly_Demand_Dataset.csv')
df['Date'] = pd.to_datetime(df['Year_Month_Str'])

# Generate full month range
all_months = pd.date_range(df['Date'].min(), df['Date'].max(), freq='MS')

# Product metadata
item_info = df.groupby('Item ID').agg(
    Category=('Category', 'first'),
    Item_Description=('Item_Description', 'first'),
    Avg_Price=('Avg_Unit_Price', 'mean'),
).reset_index()

category_map = {cat: i for i, cat in enumerate(sorted(item_info['Category'].unique()))}
item_info['Category_Code'] = item_info['Category'].map(category_map)

# Identify products with enough training history
train_df = df[df['Year_Month_Str'] < TRAIN_CUTOFF]
test_df = df[df['Year_Month_Str'] >= TRAIN_CUTOFF]

product_train_months = train_df.groupby('Item ID')['Year_Month_Str'].nunique()
test_products = set(test_df['Item ID'].unique())

# DeepAR products: enough history and present in test period
deepar_product_ids = set(
    product_train_months[product_train_months >= MIN_HISTORY_MONTHS].index
) & test_products

# Fallback products: in test but not enough history for DeepAR
fallback_product_ids = test_products - deepar_product_ids

print(f"Total products in test: {len(test_products)}")
print(f"DeepAR products ({MIN_HISTORY_MONTHS}+ months): {len(deepar_product_ids)}")
print(f"Fallback products (SMA): {len(fallback_product_ids)}")
print(f"Categories: {list(category_map.keys())}")

# 2. BUILD CONTINUOUS TIME SERIES
print("\nBuilding GluonTS datasets...")

split_date = pd.Timestamp(TRAIN_CUTOFF)

train_entries = []
test_entries = []
product_id_list = sorted(deepar_product_ids)

for item_id in product_id_list:
    product_data = df[df['Item ID'] == item_id].set_index('Date')['Total_Qty_Sold']
    info = item_info[item_info['Item ID'] == item_id].iloc[0]

    # Reindex to full continuous monthly range, fill missing months with 0
    full_series = product_data.reindex(all_months, fill_value=0.0)

    # Find first non-zero month (product launch)
    first_sale_idx = (full_series > 0).idxmax()
    full_series = full_series[first_sale_idx:]

    # Train: up to cutoff
    train_series = full_series[full_series.index < split_date]

    if len(train_series) < MIN_HISTORY_MONTHS:
        continue

    # Test: full series including test period
    test_series = full_series.copy()

    train_entries.append({
        "start": pd.Period(train_series.index[0], freq='M'),
        "target": train_series.values.astype(np.float32),
        "feat_static_cat": [int(info['Category_Code'])],
    })

    test_entries.append({
        "start": pd.Period(test_series.index[0], freq='M'),
        "target": test_series.values.astype(np.float32),
        "feat_static_cat": [int(info['Category_Code'])],
    })

train_ds = ListDataset(train_entries, freq='M')
test_ds = ListDataset(test_entries, freq='M')

print(f"DeepAR train series: {len(train_entries)}")
print(f"DeepAR test series: {len(test_entries)}")
if train_entries:
    print(f"Example series length: {len(train_entries[0]['target'])} months")

# TRAIN DEEPAR
print(f"\nTraining DeepAR ({EPOCHS} epochs)...")

# Pick the best distribution for count/demand data
try:
    from gluonts.torch.distributions import NegativeBinomialOutput
    distr = NegativeBinomialOutput()
    print("Using NegativeBinomial distribution (optimized for count data)")
except ImportError:
    try:
        from gluonts.torch.distributions import StudentTOutput
        distr = StudentTOutput()
        print("Using Student-T distribution")
    except ImportError:
        from gluonts.torch.distributions import DistributionOutput
        distr = None
        print("Using default distribution")

estimator_kwargs = dict(
    prediction_length=PREDICTION_LENGTH,
    context_length=CONTEXT_LENGTH,
    freq='M',
    num_layers=NUM_LAYERS,
    hidden_size=HIDDEN_SIZE,
    dropout_rate=DROPOUT,
    lr=LEARNING_RATE,
    batch_size=BATCH_SIZE,
    num_feat_static_cat=1,
    cardinality=[len(category_map)],
    scaling=True,
    time_features=[month_of_year],
    trainer_kwargs={
        "max_epochs": EPOCHS,
        "accelerator": "gpu" if DEVICE == "cuda" else "cpu",
        "devices": 1,
        "enable_progress_bar": True,
    },
)

if distr is not None:
    estimator_kwargs['distr_output'] = distr

estimator = DeepAREstimator(**estimator_kwargs)

predictor = estimator.train(training_data=train_ds)
print("Training complete!")

# 4. GENERATE DEEPAR FORECASTS
print("\nGenerating DeepAR forecasts...")

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,
    predictor=predictor,
    num_samples=NUM_SAMPLES,
)

forecasts = list(forecast_it)
tss = list(ts_it)

print(f"Generated {len(forecasts)} DeepAR forecasts")

# GENERATE FALLBACK FORECASTS
print(f"\n--- Generating SMA fallback for {len(fallback_product_ids)} sparse products ---")

test_months = sorted(test_df['Year_Month_Str'].unique())

def sma_forecast(history, window=3):
    if len(history) == 0:
        return 0.0
    w = min(window, len(history))
    return max(np.mean(history[-w:]), 0.0)

# COLLECT ALL PREDICTIONS
print("\nCollecting all predictions...")

def calc_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted) if len(actual) > 1 else 0.0
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100 if mask.sum() > 0 else 0.0
    return {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R²': round(r2, 4), 'MAPE_%': round(mape, 4)}

all_preds = []

# DeepAR predictions
forecast_dates = pd.date_range(start=split_date, periods=PREDICTION_LENGTH, freq='MS')

for i, (forecast, ts) in enumerate(zip(forecasts, tss)):
    item_id = product_id_list[i]
    info = item_info[item_info['Item ID'] == item_id].iloc[0]

    actual_values = ts.values[-PREDICTION_LENGTH:].flatten()
    pred_median = np.maximum(forecast.median, 0)
    pred_p10 = np.maximum(forecast.quantile(0.1), 0)
    pred_p90 = np.maximum(forecast.quantile(0.9), 0)

    for t in range(PREDICTION_LENGTH):
        if t < len(actual_values) and actual_values[t] > 0:
            all_preds.append({
                'Item ID': item_id,
                'Item_Description': info['Item_Description'],
                'Category': info['Category'],
                'Year_Month_Str': forecast_dates[t].strftime('%Y-%m'),
                'Actual': float(actual_values[t]),
                'DeepAR_Predicted': round(float(pred_median[t]), 1),
                'DeepAR_P10': round(float(pred_p10[t]), 1),
                'DeepAR_P90': round(float(pred_p90[t]), 1),
                'Method': 'DeepAR',
            })

# SMA fallback predictions
for item_id in fallback_product_ids:
    product_data = df[df['Item ID'] == item_id].sort_values('Date')
    info = item_info[item_info['Item ID'] == item_id].iloc[0]

    for test_month in test_months:
        actual_row = product_data[product_data['Year_Month_Str'] == test_month]
        if actual_row.empty:
            continue

        actual_qty = actual_row['Total_Qty_Sold'].values[0]
        history = product_data[product_data['Year_Month_Str'] < test_month]['Total_Qty_Sold'].values

        if len(history) == 0:
            continue

        pred = sma_forecast(history)

        all_preds.append({
            'Item ID': item_id,
            'Item_Description': info['Item_Description'],
            'Category': info['Category'],
            'Year_Month_Str': test_month,
            'Actual': float(actual_qty),
            'DeepAR_Predicted': round(pred, 1),
            'DeepAR_P10': round(pred * 0.5, 1),
            'DeepAR_P90': round(pred * 1.5, 1),
            'Method': 'SMA_Fallback',
        })

deepar_df = pd.DataFrame(all_preds)

deepar_only = deepar_df[deepar_df['Method'] == 'DeepAR']
sma_only = deepar_df[deepar_df['Method'] == 'SMA_Fallback']

print(f"DeepAR predictions: {len(deepar_only)}")
print(f"SMA fallback predictions: {len(sma_only)}")
print(f"Total predictions: {len(deepar_df)}")

# EVALUATE
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# DeepAR metrics
da_metrics = calc_metrics(deepar_only['Actual'].values, deepar_only['DeepAR_Predicted'].values)
print(f"\nDeepAR Only ({len(deepar_only)} predictions on {len(deepar_product_ids)} products):")
for k, v in da_metrics.items():
    print(f"  {k}: {v}")

# Combined metrics
combined_metrics = calc_metrics(deepar_df['Actual'].values, deepar_df['DeepAR_Predicted'].values)
print(f"\nCombined DeepAR + SMA Fallback ({len(deepar_df)} predictions):")
for k, v in combined_metrics.items():
    print(f"  {k}: {v}")

# LOAD BASELINES FOR 3-WAY COMPARISON
print("\nLoading baselines...")
bl_path = Path('../data/results/baseline ARIMA/baseline_predictions.csv')
gb_path = Path('../data/results/baseline ML/test_predictions.csv')

comparison_data = deepar_df.copy()

if bl_path.exists():
    bl_df = pd.read_csv(bl_path)
    comparison_data = comparison_data.merge(
        bl_df[['Item ID', 'Year_Month_Str', 'Baseline_Predicted']],
        on=['Item ID', 'Year_Month_Str'], how='left'
    )

if gb_path.exists():
    gb_df = pd.read_csv(gb_path)
    comparison_data = comparison_data.merge(
        gb_df[['Item ID', 'Year_Month_Str', 'Predicted']].rename(columns={'Predicted': 'GB_Predicted'}),
        on=['Item ID', 'Year_Month_Str'], how='left'
    )

fair_comp = comparison_data.dropna(subset=['Baseline_Predicted', 'GB_Predicted']).copy()
print(f"  Fair comparison rows: {len(fair_comp)}")

# DeepAR-only fair comparison
fair_deepar = fair_comp[fair_comp['Method'] == 'DeepAR'].copy()
print(f"  Fair DeepAR-only rows: {len(fair_deepar)}")


# THREE-WAY COMPARISON
print("3-WAY MODEL COMPARISON...")


if len(fair_deepar) > 0:
    bl_m = calc_metrics(fair_deepar['Actual'].values, fair_deepar['Baseline_Predicted'].values)
    gb_m = calc_metrics(fair_deepar['Actual'].values, fair_deepar['GB_Predicted'].values)
    da_m = calc_metrics(fair_deepar['Actual'].values, fair_deepar['DeepAR_Predicted'].values)

    print(f"\nOn DeepAR-eligible products only ({len(fair_deepar)} predictions):")
    print(f"  {'Model':<25} {'MAE':>8} {'RMSE':>8} {'R²':>8} {'MAPE':>8}")
    print(f"  {'-' * 57}")
    print(f"  {'ARIMA+SMA (Classical)':<25} {bl_m['MAE']:>8.2f} {bl_m['RMSE']:>8.2f} {bl_m['R²']:>8.4f} {bl_m['MAPE_%']:>7.1f}%")
    print(f"  {'Gradient Boosting (ML)':<25} {gb_m['MAE']:>8.2f} {gb_m['RMSE']:>8.2f} {gb_m['R²']:>8.4f} {gb_m['MAPE_%']:>7.1f}%")
    print(f"  {'DeepAR (Proposed)':<25} {da_m['MAE']:>8.2f} {da_m['RMSE']:>8.2f} {da_m['R²']:>8.4f} {da_m['MAPE_%']:>7.1f}%")

    comp_table = pd.DataFrame({
        'ARIMA+SMA (Classical)': bl_m,
        'Gradient Boosting (ML)': gb_m,
        'DeepAR (Proposed)': da_m
    }).round(4)
    comp_table.to_csv(out / 'three_way_comparison.csv')

    # Category breakdown
    cat_results = []
    for cat in sorted(fair_deepar['Category'].unique()):
        sub = fair_deepar[fair_deepar['Category'] == cat]
        if len(sub) < 2:
            continue
        cat_results.append({
            'Category': cat, 'N': len(sub),
            'ARIMA_MAE': calc_metrics(sub['Actual'].values, sub['Baseline_Predicted'].values)['MAE'],
            'GB_MAE': calc_metrics(sub['Actual'].values, sub['GB_Predicted'].values)['MAE'],
            'DeepAR_MAE': calc_metrics(sub['Actual'].values, sub['DeepAR_Predicted'].values)['MAE'],
            'ARIMA_R²': calc_metrics(sub['Actual'].values, sub['Baseline_Predicted'].values)['R²'],
            'GB_R²': calc_metrics(sub['Actual'].values, sub['GB_Predicted'].values)['R²'],
            'DeepAR_R²': calc_metrics(sub['Actual'].values, sub['DeepAR_Predicted'].values)['R²'],
        })

    cat_df = pd.DataFrame(cat_results).round(4)
    cat_df.to_csv(out / 'category_three_way_comparison.csv', index=False)

    print(f"\nCategory Breakdown:")
    print(cat_df[['Category', 'N', 'ARIMA_MAE', 'GB_MAE', 'DeepAR_MAE']].to_string(index=False))

# CHARTS
print("\nGenerating charts...")

# 3-Way Bar Comparison
if len(fair_deepar) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    models_names = ['ARIMA+SMA', 'Gradient\nBoosting', 'DeepAR']
    colors = ['#C44E52', '#4C72B0', '#55A868']

    for idx, (metric, vals) in enumerate([
        ('MAE', [bl_m['MAE'], gb_m['MAE'], da_m['MAE']]),
        ('RMSE', [bl_m['RMSE'], gb_m['RMSE'], da_m['RMSE']]),
        ('R²', [bl_m['R²'], gb_m['R²'], da_m['R²']])
    ]):
        bars = axes[idx].bar(models_names, vals, color=colors, width=0.5)
        axes[idx].set_title(metric, fontsize=14, fontweight='bold')
        for bar, val in zip(bars, vals):
            axes[idx].text(bar.get_x() + bar.get_width()/2,
                          bar.get_height() + 0.01 * max(abs(v) for v in vals),
                          f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('3-Way Model Comparison — DeepAR-Eligible Products', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out / 'three_way_comparison.png', dpi=150)
    plt.close()

# Actual vs Predicted — DeepAR
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(deepar_only['Actual'], deepar_only['DeepAR_Predicted'], alpha=0.3, s=15, color='#55A868')
lim = min(200, max(deepar_only['Actual'].max(), deepar_only['DeepAR_Predicted'].max()))
ax.plot([0, lim], [0, lim], 'r--', linewidth=2, label='Perfect Prediction')
ax.set_xlim(0, lim); ax.set_ylim(0, lim)
ax.set_xlabel('Actual Qty Sold', fontsize=12)
ax.set_ylabel('Predicted Qty Sold', fontsize=12)
ax.set_title(f'Actual vs Predicted — DeepAR\n(R² = {da_metrics["R²"]:.4f})', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(out / 'deepar_actual_vs_predicted.png', dpi=150)
plt.close()

# Side-by-Side All 3 Models
if len(fair_deepar) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    lim = 150
    model_data = [
        ('ARIMA+SMA', fair_deepar['Baseline_Predicted'], '#C44E52', bl_m['R²']),
        ('Gradient Boosting', fair_deepar['GB_Predicted'], '#4C72B0', gb_m['R²']),
        ('DeepAR', fair_deepar['DeepAR_Predicted'], '#55A868', da_m['R²']),
    ]
    for idx, (name, preds, color, r2) in enumerate(model_data):
        axes[idx].scatter(fair_deepar['Actual'], preds, alpha=0.3, s=15, color=color)
        axes[idx].plot([0, lim], [0, lim], 'k--', linewidth=2)
        axes[idx].set_xlim(0, lim); axes[idx].set_ylim(0, lim)
        axes[idx].set_xlabel('Actual'); axes[idx].set_ylabel('Predicted')
        axes[idx].set_title(f'{name}\nR² = {r2:.4f}', fontsize=12, fontweight='bold')

    plt.suptitle('Actual vs Predicted — All 3 Models (DeepAR-Eligible Products)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out / 'three_way_scatter.png', dpi=150)
    plt.close()

# Category 3-way
if len(fair_deepar) > 0 and len(cat_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cat_sorted = cat_df.sort_values('DeepAR_MAE', ascending=True)
    y_pos = np.arange(len(cat_sorted))
    bar_h = 0.25

    axes[0].barh(y_pos - bar_h, cat_sorted['ARIMA_MAE'], bar_h, color='#C44E52', label='ARIMA+SMA')
    axes[0].barh(y_pos, cat_sorted['GB_MAE'], bar_h, color='#4C72B0', label='Gradient Boosting')
    axes[0].barh(y_pos + bar_h, cat_sorted['DeepAR_MAE'], bar_h, color='#55A868', label='DeepAR')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(cat_sorted['Category'], fontsize=9)
    axes[0].set_title('MAE by Category', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=9)

    axes[1].barh(y_pos - bar_h, cat_sorted['ARIMA_R²'], bar_h, color='#C44E52', label='ARIMA+SMA')
    axes[1].barh(y_pos, cat_sorted['GB_R²'], bar_h, color='#4C72B0', label='Gradient Boosting')
    axes[1].barh(y_pos + bar_h, cat_sorted['DeepAR_R²'], bar_h, color='#55A868', label='DeepAR')
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels(cat_sorted['Category'], fontsize=9)
    axes[1].set_title('R² by Category', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)

    plt.suptitle('Category Performance — 3-Way', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out / 'category_three_way.png', dpi=150)
    plt.close()

# Sample forecasts with prediction intervals
top_products = deepar_only.groupby('Item ID')['Actual'].sum().nlargest(6).index.tolist()

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes_flat = axes.flatten()

for idx, item_id in enumerate(top_products):
    ax = axes_flat[idx]
    item_data = deepar_only[deepar_only['Item ID'] == item_id].sort_values('Year_Month_Str')
    full_hist = df[df['Item ID'] == item_id].sort_values('Year_Month_Str')

    ax.plot(pd.to_datetime(full_hist['Year_Month_Str']), full_hist['Total_Qty_Sold'],
            'b-', linewidth=1.5, label='History', alpha=0.7)

    fc_dates = pd.to_datetime(item_data['Year_Month_Str'])
    ax.plot(fc_dates, item_data['Actual'], 'ko-', markersize=5, linewidth=2, label='Actual')
    ax.plot(fc_dates, item_data['DeepAR_Predicted'], 'g^-', markersize=5, linewidth=2, label='DeepAR')
    ax.fill_between(fc_dates, item_data['DeepAR_P10'], item_data['DeepAR_P90'],
                    alpha=0.2, color='green', label='80% PI')
    ax.axvline(pd.Timestamp(TRAIN_CUTOFF), color='red', linestyle='--', alpha=0.5)

    desc = item_data['Item_Description'].iloc[0] if len(item_data) > 0 else item_id
    ax.set_title(f'{str(desc)[:45]}', fontsize=9, fontweight='bold')
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    if idx == 0:
        ax.legend(fontsize=7)

plt.suptitle('Sample Forecasts with 80% Prediction Intervals', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out / 'sample_forecasts_with_intervals.png', dpi=150)
plt.close()

# SAVE RESULTS
deepar_df.to_csv(out / 'deepar_predictions.csv', index=False)
if len(fair_deepar) > 0:
    fair_deepar.to_csv(out / 'three_way_predictions.csv', index=False)

config = {
    'prediction_length': PREDICTION_LENGTH,
    'context_length': CONTEXT_LENGTH,
    'min_history_months': MIN_HISTORY_MONTHS,
    'epochs': EPOCHS,
    'learning_rate': LEARNING_RATE,
    'batch_size': BATCH_SIZE,
    'num_layers': NUM_LAYERS,
    'hidden_size': HIDDEN_SIZE,
    'dropout': DROPOUT,
    'device': DEVICE,
    'num_deepar_products': len(deepar_product_ids),
    'num_fallback_products': len(fallback_product_ids),
    'num_categories': len(category_map),
    'category_map': category_map,
}
with open(out / 'deepar_config.json', 'w') as f:
    json.dump(config, f, indent=2, default=str)

# SUMMARY
print("DEEPAR TRAINING COMPLETE")
print(f"\nDeepAR Results ({len(deepar_product_ids)} products, {len(deepar_only)} predictions):")
print(f"  MAE:  {da_metrics['MAE']:.2f}")
print(f"  RMSE: {da_metrics['RMSE']:.2f}")
print(f"  R²:   {da_metrics['R²']:.4f}")
print(f"  MAPE: {da_metrics['MAPE_%']:.1f}%")

if len(fair_deepar) > 0:
    print(f"\n3-Way Comparison (same products, same test period):")
    print(comp_table.to_string())

print(f"\nFiles saved to: {out}/")
