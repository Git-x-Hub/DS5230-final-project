import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

out = Path('../data/results/baseline ARIMA')
out.mkdir(exist_ok=True)

# ARIMA
class SimpleARIMA:

    def __init__(self):
        self.ar1 = 0.0
        self.ma1 = 0.0
        self.const = 0.0
        self.fitted = False

    def _difference(self, series):
        return np.diff(series)

    def _neg_log_likelihood(self, params, diff_y):
        ar1, ma1, const, sigma = params
        sigma = max(sigma, 1e-6)
        n = len(diff_y)
        residuals = np.zeros(n)

        for t in range(n):
            pred = const
            if t >= 1:
                pred += ar1 * diff_y[t-1]
                pred += ma1 * residuals[t-1]
            residuals[t] = diff_y[t] - pred

        ll = -0.5 * n * np.log(2 * np.pi * sigma**2) - 0.5 * np.sum(residuals**2) / sigma**2
        return -ll

    def fit(self, series):
        series = np.array(series, dtype=float)
        self.original = series
        diff_y = self._difference(series)
        self.diff_y = diff_y

        if len(diff_y) < 3:
            self.fitted = False
            return self

        x0 = [0.1, 0.1, np.mean(diff_y), np.std(diff_y) + 1e-6]
        bounds = [(-0.99, 0.99), (-0.99, 0.99), (-100, 100), (1e-6, None)]

        try:
            result = minimize(self._neg_log_likelihood, x0, args=(diff_y,),
                            method='L-BFGS-B', bounds=bounds,
                            options={'maxiter': 200})
            self.ar1, self.ma1, self.const, self.sigma = result.x

            n = len(diff_y)
            self.residuals = np.zeros(n)
            for t in range(n):
                pred = self.const
                if t >= 1:
                    pred += self.ar1 * diff_y[t-1]
                    pred += self.ma1 * self.residuals[t-1]
                self.residuals[t] = diff_y[t] - pred

            self.fitted = True
        except:
            self.fitted = False

        return self

    def forecast(self, steps=1):
        if not self.fitted:
            return np.full(steps, self.original[-1])

        preds = []
        last_val = self.original[-1]
        last_diff = self.diff_y[-1]
        last_resid = self.residuals[-1]

        for s in range(steps):
            diff_pred = self.const + self.ar1 * last_diff + self.ma1 * last_resid
            val_pred = last_val + diff_pred
            preds.append(max(val_pred, 0))
            last_diff = diff_pred
            last_resid = 0
            last_val = val_pred

        return np.array(preds)

# SIMPLE MOVING AVERAGE
def sma_forecast(series, window=3, steps=1):
    """Simple Moving Average forecast."""
    series = np.array(series, dtype=float)
    if len(series) < window:
        window = len(series)
    avg = np.mean(series[-window:])
    return np.full(steps, max(avg, 0))


# EVALUATION
def calc_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    mask = actual > 0
    mape = np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100
    return {'MAE': mae, 'RMSE': rmse, 'R²': r2, 'MAPE_%': mape}

# LOAD DATA
df = pd.read_csv('../data/processed/Monthly_Demand_Dataset.csv')
df['Date'] = pd.to_datetime(df['Year_Month_Str'])

train_cutoff = '2025-07'
test_months = sorted(df[df['Year_Month_Str'] >= train_cutoff]['Year_Month_Str'].unique())

print(f"Test period: {test_months[0]} to {test_months[-1]} ({len(test_months)} months)")

# Categorize products by history length
product_history = df[df['Year_Month_Str'] < train_cutoff].groupby('Item ID')['Year_Month_Str'].nunique()
arima_products = set(product_history[product_history >= 18].index)
sma_products = set(product_history[(product_history >= 3) & (product_history < 18)].index)
too_short = set(product_history[product_history < 3].index)

print(f"\nProduct split:")
print(f"  ARIMA (18+ months):    {len(arima_products)} products")
print(f"  SMA (3-17 months):     {len(sma_products)} products")
print(f"  Too short (<3 months): {len(too_short)} products (will use last value)")

# GENERATE FORECASTS — MONTH BY MONTH (ROLLING)
all_predictions = []
arima_fit_count = 0
arima_fail_count = 0
sma_count = 0
naive_count = 0

all_products = df['Item ID'].unique()
total = len(all_products)

for idx, product_id in enumerate(all_products):
    if (idx + 1) % 100 == 0:
        print(f"  Processing product {idx+1}/{total}...")

    product_df = df[df['Item ID'] == product_id].sort_values('Date')
    product_desc = product_df['Item_Description'].iloc[0]
    category = product_df['Category'].iloc[0]

    for test_month in test_months:
        actual_row = product_df[product_df['Year_Month_Str'] == test_month]
        if actual_row.empty:
            continue

        actual_qty = actual_row['Total_Qty_Sold'].values[0]
        history = product_df[product_df['Year_Month_Str'] < test_month]['Total_Qty_Sold'].values

        if len(history) == 0:
            continue

        if product_id in arima_products and len(history) >= 18:
            method = 'ARIMA(1,1,1)'
            model = SimpleARIMA()
            model.fit(history)
            pred = model.forecast(steps=1)[0]
            if model.fitted:
                arima_fit_count += 1
            else:
                arima_fail_count += 1
                method = 'ARIMA→Naive'
        elif len(history) >= 3:
            method = 'SMA(3)'
            pred = sma_forecast(history, window=3, steps=1)[0]
            sma_count += 1
        else:
            method = 'Naive'
            pred = history[-1]
            naive_count += 1

        all_predictions.append({
            'Item ID': product_id,
            'Item_Description': product_desc,
            'Category': category,
            'Year_Month_Str': test_month,
            'Actual': actual_qty,
            'Baseline_Predicted': round(pred, 1),
            'Method': method
        })

baseline_df = pd.DataFrame(all_predictions)
print(f"\nForecasting complete!")
print(f"  ARIMA fitted: {arima_fit_count} | ARIMA fallback: {arima_fail_count}")
print(f"  SMA: {sma_count} | Naive: {naive_count} | Total: {len(baseline_df)}")

# EVALUATE BASELINE
overall = calc_metrics(baseline_df['Actual'].values, baseline_df['Baseline_Predicted'].values)

print("\n" + "="*70)
print("HYBRID BASELINE RESULTS (ARIMA + SMA)")
print("="*70)
for k, v in overall.items():
    print(f"  {k}: {v:.4f}" if 'R' in k else f"  {k}: {v:.2f}")

print("\nBy Method:")
for method in baseline_df['Method'].unique():
    subset = baseline_df[baseline_df['Method'] == method]
    m = calc_metrics(subset['Actual'].values, subset['Baseline_Predicted'].values)
    print(f"  {method:20s} | N={len(subset):5d} | MAE={m['MAE']:.2f} | R²={m['R²']:.4f} | MAPE={m['MAPE_%']:.1f}%")

print("\nBy Category:")
for cat in sorted(baseline_df['Category'].unique()):
    subset = baseline_df[baseline_df['Category'] == cat]
    m = calc_metrics(subset['Actual'].values, subset['Baseline_Predicted'].values)
    print(f"  {cat:20s} | N={len(subset):4d} | MAE={m['MAE']:.2f} | R²={m['R²']:.4f}")

# COMPARE WITH GRADIENT BOOSTING
gb_path = Path('../data/results/baseline ML/test_predictions.csv')
if gb_path.exists():
    gb_preds = pd.read_csv(gb_path)
    comparison = baseline_df.merge(
        gb_preds[['Item ID', 'Year_Month_Str', 'Predicted']].rename(columns={'Predicted': 'GB_Predicted'}),
        on=['Item ID', 'Year_Month_Str'], how='inner'
    )

    gb_metrics = calc_metrics(comparison['Actual'].values, comparison['GB_Predicted'].values)
    bl_metrics = calc_metrics(comparison['Actual'].values, comparison['Baseline_Predicted'].values)

    print(f"\n{'='*70}")
    print("HEAD-TO-HEAD: BASELINE vs GRADIENT BOOSTING")
    print("="*70)
    comp_table = pd.DataFrame({'Baseline (ARIMA+SMA)': bl_metrics, 'Gradient Boosting': gb_metrics}).round(4)
    print(comp_table.to_string())

    improvement = {
        'MAE': (bl_metrics['MAE'] - gb_metrics['MAE']) / bl_metrics['MAE'] * 100,
        'RMSE': (bl_metrics['RMSE'] - gb_metrics['RMSE']) / bl_metrics['RMSE'] * 100,
        'MAPE': (bl_metrics['MAPE_%'] - gb_metrics['MAPE_%']) / bl_metrics['MAPE_%'] * 100,
    }
    print(f"\nGB improvement: MAE {improvement['MAE']:.1f}% | RMSE {improvement['RMSE']:.1f}% | MAPE {improvement['MAPE']:.1f}%")

    comparison.to_csv(out / 'baseline_vs_gb_comparison.csv', index=False)
    comp_table.to_csv(out / 'model_comparison_summary.csv')

    # Category comparison
    cat_comp_rows = []
    for cat in sorted(comparison['Category'].unique()):
        sub = comparison[comparison['Category'] == cat]
        bl_m = calc_metrics(sub['Actual'].values, sub['Baseline_Predicted'].values)
        gb_m = calc_metrics(sub['Actual'].values, sub['GB_Predicted'].values)
        cat_comp_rows.append({
            'Category': cat,
            'Baseline_MAE': bl_m['MAE'], 'GB_MAE': gb_m['MAE'],
            'Baseline_R²': bl_m['R²'], 'GB_R²': gb_m['R²'],
            'Baseline_MAPE': bl_m['MAPE_%'], 'GB_MAPE': gb_m['MAPE_%'],
            'MAE_Improvement_%': (bl_m['MAE'] - gb_m['MAE']) / bl_m['MAE'] * 100 if bl_m['MAE'] > 0 else 0
        })
    cat_comp_df = pd.DataFrame(cat_comp_rows).round(4)
    cat_comp_df.to_csv(out / 'category_comparison.csv', index=False)
else:
    comparison = None
    print("\nNote: data/results/baseline ML/test_predictions.csv not found — skip GB comparison.")


# 5. SAVE BASELINE PREDICTIONS
baseline_df.to_csv(out / 'baseline_predictions.csv', index=False)

# HEAD-TO-HEAD BAR CHART
if comparison is not None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics_names = ['MAE', 'RMSE', 'R²']
    bl_vals = [bl_metrics['MAE'], bl_metrics['RMSE'], bl_metrics['R²']]
    gb_vals = [gb_metrics['MAE'], gb_metrics['RMSE'], gb_metrics['R²']]

    for idx, (metric, bv, gv) in enumerate(zip(metrics_names, bl_vals, gb_vals)):
        x = np.arange(2)
        bars = axes[idx].bar(x, [bv, gv], color=['#C44E52', '#4C72B0'], width=0.5)
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels(['ARIMA+SMA\n(Baseline)', 'Gradient\nBoosting'], fontsize=10)
        axes[idx].set_title(metric, fontsize=14, fontweight='bold')
        for bar, val in zip(bars, [bv, gv]):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01*max(bv,gv),
                          f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.suptitle('Baseline vs Gradient Boosting — Overall', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out / 'baseline_vs_gb_overall.png', dpi=150)
    plt.close()

    # CATEGORY-LEVEL COMPARISON
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    cat_comp_sorted = cat_comp_df.sort_values('Baseline_MAE', ascending=True)
    y_pos = np.arange(len(cat_comp_sorted))
    bar_h = 0.35

    axes[0].barh(y_pos - bar_h/2, cat_comp_sorted['Baseline_MAE'], bar_h, color='#C44E52', label='Baseline')
    axes[0].barh(y_pos + bar_h/2, cat_comp_sorted['GB_MAE'], bar_h, color='#4C72B0', label='Gradient Boosting')
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(cat_comp_sorted['Category'], fontsize=9)
    axes[0].set_title('MAE by Category', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('MAE')
    axes[0].legend()

    cat_comp_sorted2 = cat_comp_df.sort_values('GB_R²', ascending=True)
    y_pos2 = np.arange(len(cat_comp_sorted2))
    axes[1].barh(y_pos2 - bar_h/2, cat_comp_sorted2['Baseline_R²'], bar_h, color='#C44E52', label='Baseline')
    axes[1].barh(y_pos2 + bar_h/2, cat_comp_sorted2['GB_R²'], bar_h, color='#4C72B0', label='Gradient Boosting')
    axes[1].set_yticks(y_pos2)
    axes[1].set_yticklabels(cat_comp_sorted2['Category'], fontsize=9)
    axes[1].set_title('R² by Category', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('R²')
    axes[1].legend()

    plt.suptitle('Category Comparison — Baseline vs Gradient Boosting', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out / 'category_comparison.png', dpi=150)
    plt.close()

    # ACTUAL VS PREDICTED
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lim = 150

    axes[0].scatter(baseline_df['Actual'], baseline_df['Baseline_Predicted'], alpha=0.3, s=15, color='#C44E52')
    axes[0].plot([0, lim], [0, lim], 'k--', linewidth=2)
    axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
    axes[0].set_xlabel('Actual'); axes[0].set_ylabel('Predicted')
    axes[0].set_title(f'Baseline (ARIMA+SMA)\nR² = {overall["R²"]:.4f}', fontsize=12, fontweight='bold')

    axes[1].scatter(comparison['Actual'], comparison['GB_Predicted'], alpha=0.3, s=15, color='#4C72B0')
    axes[1].plot([0, lim], [0, lim], 'k--', linewidth=2)
    axes[1].set_xlim(0, lim); axes[1].set_ylim(0, lim)
    axes[1].set_xlabel('Actual'); axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Gradient Boosting\nR² = {gb_metrics["R²"]:.4f}', fontsize=12, fontweight='bold')

    plt.suptitle('Actual vs Predicted — Side by Side', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out / 'actual_vs_predicted_comparison.png', dpi=150)
    plt.close()


# METHOD BREAKDOWN
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

method_counts = baseline_df['Method'].value_counts()
colors_pie = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
axes[0].pie(method_counts.values, labels=method_counts.index, autopct='%1.1f%%',
           colors=colors_pie[:len(method_counts)], startangle=90)
axes[0].set_title('Forecast Method Distribution', fontsize=13, fontweight='bold')

method_perf = []
for method in baseline_df['Method'].unique():
    sub = baseline_df[baseline_df['Method'] == method]
    m = calc_metrics(sub['Actual'].values, sub['Baseline_Predicted'].values)
    method_perf.append({'Method': method, 'MAE': m['MAE'], 'R²': m['R²'], 'Count': len(sub)})

mp_df = pd.DataFrame(method_perf).sort_values('MAE')
y_pos = np.arange(len(mp_df))
axes[1].barh(y_pos, mp_df['MAE'], color=colors_pie[:len(mp_df)])
axes[1].set_yticks(y_pos)
axes[1].set_yticklabels([f"{row['Method']}\n(n={row['Count']}, R²={row['R²']:.3f})"
                          for _, row in mp_df.iterrows()], fontsize=9)
axes[1].set_xlabel('MAE')
axes[1].set_title('Performance by Method', fontsize=13, fontweight='bold')
for i, v in enumerate(mp_df['MAE']):
    axes[1].text(v + 0.1, i, f'{v:.2f}', va='center', fontsize=11)

plt.suptitle('Baseline Method Breakdown', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out / 'method_breakdown.png', dpi=150)
plt.close()
