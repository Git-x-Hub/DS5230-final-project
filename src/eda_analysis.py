import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.bbox'] = 'tight'

df = pd.read_csv('../data/processed/Monthly_Demand_Dataset.csv')
df['Date'] = pd.to_datetime(df['Year_Month_Str'])

out = Path('../data/results/figures')
out.mkdir(exist_ok=True)


# TARGET DISTRIBUTION

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['Total_Qty_Sold'], bins=80, color='#4C72B0', edgecolor='white')
axes[0].set_title('Distribution of Monthly Qty Sold (Raw)')
axes[0].set_xlabel('Total Qty Sold')
axes[0].set_ylabel('Frequency')
axes[0].axvline(df['Total_Qty_Sold'].median(), color='red', ls='--', label=f"Median: {df['Total_Qty_Sold'].median():.0f}")
axes[0].axvline(df['Total_Qty_Sold'].mean(), color='orange', ls='--', label=f"Mean: {df['Total_Qty_Sold'].mean():.1f}")
axes[0].legend()

axes[1].hist(np.log1p(df['Total_Qty_Sold']), bins=50, color='#55A868', edgecolor='white')
axes[1].set_title('Distribution of log(1 + Qty Sold)')
axes[1].set_xlabel('log(1 + Total Qty Sold)')
axes[1].set_ylabel('Frequency')

plt.suptitle('Target Variable Distribution', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out / 'target_distribution.png')
plt.close()


# OVERALL DEMAND TREND (monthly)
monthly_total = df.groupby('Date').agg(
    Total_Qty=('Total_Qty_Sold', 'sum'),
    Total_Revenue=('Total_Revenue', 'sum'),
    Active_Products=('Item ID', 'nunique'),
    Total_Orders=('Num_Orders', 'sum')
).reset_index()

fig, ax1 = plt.subplots(figsize=(14, 5))
ax1.bar(monthly_total['Date'], monthly_total['Total_Qty'], width=25, color='#4C72B0', alpha=0.7, label='Total Qty Sold')
ax1.set_ylabel('Total Quantity Sold', color='#4C72B0')
ax1.tick_params(axis='y', labelcolor='#4C72B0')

ax2 = ax1.twinx()
ax2.plot(monthly_total['Date'], monthly_total['Total_Revenue'], color='#C44E52', linewidth=2, label='Total Revenue')
ax2.set_ylabel('Total Revenue (₱)', color='#C44E52')
ax2.tick_params(axis='y', labelcolor='#C44E52')
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₱{x:,.0f}'))

plt.title('Monthly Demand & Revenue Trend', fontsize=14, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.95))
plt.tight_layout()
plt.savefig(out / 'monthly_demand_trend.png')
plt.close()

# DEMAND BY CATEGORY
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

cat_qty = df.groupby('Category')['Total_Qty_Sold'].sum().sort_values(ascending=True)
cat_qty.plot(kind='barh', ax=axes[0], color='#4C72B0')
axes[0].set_title('Total Qty Sold by Category')
axes[0].set_xlabel('Total Qty Sold')
for i, v in enumerate(cat_qty):
    axes[0].text(v + 50, i, f'{v:,.0f}', va='center', fontsize=9)

cat_rev = df.groupby('Category')['Total_Revenue'].sum().sort_values(ascending=True)
cat_rev.plot(kind='barh', ax=axes[1], color='#55A868')
axes[1].set_title('Total Revenue by Category')
axes[1].set_xlabel('Total Revenue (₱)')
axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'₱{x/1e6:.1f}M'))
for i, v in enumerate(cat_rev):
    axes[1].text(v + 5000, i, f'₱{v:,.0f}', va='center', fontsize=8)

plt.suptitle('Category Performance', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(out / 'category_performance.png')
plt.close()


# SEASONALITY — MONTH-OF-YEAR PATTERN
month_pattern = df.groupby('Month')['Total_Qty_Sold'].agg(['mean', 'median', 'sum']).reset_index()

fig, ax = plt.subplots(figsize=(10, 5))
months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
ax.bar(month_pattern['Month'], month_pattern['sum'], color='#4C72B0', alpha=0.5, label='Total Qty')
ax2 = ax.twinx()
ax2.plot(month_pattern['Month'], month_pattern['mean'], 'o-', color='#C44E52', linewidth=2, label='Avg Qty per Product')
ax.set_xticks(range(1,13))
ax.set_xticklabels(months)
ax.set_ylabel('Total Qty Sold', color='#4C72B0')
ax2.set_ylabel('Avg Qty per Product-Month', color='#C44E52')
plt.title('Seasonality — Month-of-Year Demand Pattern', fontsize=14, fontweight='bold')
fig.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.tight_layout()
plt.savefig(out / 'seasonality.png')
plt.close()

# CATEGORY SEASONALITY HEATMAP
cat_month = df.groupby(['Category', 'Month'])['Total_Qty_Sold'].mean().reset_index()
cat_month_pivot = cat_month.pivot(index='Category', columns='Month', values='Total_Qty_Sold')
cat_month_pivot.columns = months

fig, ax = plt.subplots(figsize=(12, 6))
sns.heatmap(cat_month_pivot, annot=True, fmt='.1f', cmap='YlOrRd', ax=ax, linewidths=0.5)
plt.title('Avg Monthly Demand by Category (Seasonality Heatmap)', fontsize=14, fontweight='bold')
plt.ylabel('')
plt.tight_layout()
plt.savefig(out / 'category_seasonality_heatmap.png')
plt.close()


# TOP 15 PRODUCTS BY TOTAL QUANTITY
top15 = df.groupby(['Item ID', 'Item_Description', 'Category']).agg(
    Total_Qty=('Total_Qty_Sold', 'sum'),
    Avg_Monthly_Qty=('Total_Qty_Sold', 'mean'),
    Months_Active=('Year_Month_Str', 'nunique')
).reset_index().nlargest(15, 'Total_Qty')

fig, ax = plt.subplots(figsize=(12, 7))
labels = [f"{row['Item_Description'][:40]}\n({row['Category']})" for _, row in top15.iterrows()]
bars = ax.barh(range(len(top15)), top15['Total_Qty'].values, color='#4C72B0')
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(labels, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Total Qty Sold (All Time)')
for i, v in enumerate(top15['Total_Qty'].values):
    ax.text(v + 20, i, f'{v:,.0f}', va='center', fontsize=9)
plt.title('Top 15 Products by Total Quantity Sold', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out / 'top15_products.png')
plt.close()

# FEATURE CORRELATION WITH TARGET
numeric_cols = [
    'Total_Qty_Sold', 'Num_Orders', 'Num_Days_With_Sales', 'Avg_Qty_Per_Order',
    'Avg_Unit_Price', 'Median_Unit_Price', 'Min_Unit_Price', 'Max_Unit_Price',
    'Std_Unit_Price', 'Price_Change_Pct', 'Total_Revenue',
    'Qty_Lag_1', 'Qty_Lag_2', 'Qty_Lag_3',
    'Qty_Rolling_3M_Avg', 'Qty_Rolling_6M_Avg', 'Qty_Growth_Pct',
    'Lifetime_Total_Qty', 'Lifetime_Avg_Price', 'Months_Since_Launch', 'Month'
]

corr = df[numeric_cols].corr()
target_corr = corr['Total_Qty_Sold'].drop('Total_Qty_Sold').sort_values()

fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#C44E52' if v < 0 else '#4C72B0' for v in target_corr.values]
target_corr.plot(kind='barh', ax=ax, color=colors)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_title('Feature Correlation with Total_Qty_Sold', fontsize=14, fontweight='bold')
ax.set_xlabel('Pearson Correlation')
for i, v in enumerate(target_corr.values):
    ax.text(v + 0.01 if v >= 0 else v - 0.06, i, f'{v:.2f}', va='center', fontsize=9)
plt.tight_layout()
plt.savefig(out / 'feature_correlation.png')
plt.close()

# FULL CORRELATION HEATMAP (top features)
top_features = ['Total_Qty_Sold', 'Qty_Lag_1', 'Qty_Rolling_3M_Avg', 'Qty_Rolling_6M_Avg',
                'Num_Orders', 'Num_Days_With_Sales', 'Total_Revenue', 'Lifetime_Total_Qty',
                'Avg_Unit_Price', 'Months_Since_Launch', 'Month']

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[top_features].corr(), annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=ax, linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap — Key Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(out / 'correlation_heatmap.png')
plt.close()

print("EDA SUMMARY")
print("="*60)
print(f"\nDataset: {len(df):,} rows | {df['Item ID'].nunique()} products | {df['Year_Month_Str'].nunique()} months")
print(f"\nTarget (Total_Qty_Sold):")
print(f"  Mean: {df['Total_Qty_Sold'].mean():.1f} | Median: {df['Total_Qty_Sold'].median():.0f} | Std: {df['Total_Qty_Sold'].std():.1f}")
print(f"  Min: {df['Total_Qty_Sold'].min()} | Max: {df['Total_Qty_Sold'].max()} | Skewness: {df['Total_Qty_Sold'].skew():.2f}")

print(f"\nTop correlated features with target:")
for feat, val in target_corr.sort_values(ascending=False).head(5).items():
    print(f"  {feat}: {val:.3f}")

print(f"\nProducts with <6 months history: {(df.groupby('Item ID')['Year_Month_Str'].nunique() < 6).sum()}")
print(f"Products with <3 months history: {(df.groupby('Item ID')['Year_Month_Str'].nunique() < 3).sum()}")

zero_months = (df['Total_Qty_Sold'] == 0).sum()
print(f"\nZero-demand months: {zero_months} ({zero_months/len(df)*100:.1f}%)")
