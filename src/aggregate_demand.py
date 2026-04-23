import pandas as pd
import numpy as np


# LOAD & CLEAN
df = pd.read_csv('../data/processed/Categorized_Data.csv')

# Parse dates
df['Date'] = pd.to_datetime(df['Date Order Made'], format='mixed', errors='coerce', dayfirst=False)

# Drop rows with null Item ID or unparseable dates
df = df.dropna(subset=['Item ID', 'Date']).copy()

# CREATE TIME FEATURES

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Year_Month'] = df['Date'].dt.to_period('M')
df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
df['Day_of_Week'] = df['Date'].dt.dayofweek          # 0=Mon, 6=Sun
df['Day_of_Month'] = df['Date'].dt.day
df['Is_Weekend'] = (df['Day_of_Week'] >= 5).astype(int)
df['Revenue'] = df['Quantity'] * df['Unit Price When Sold']

# MONTHLY AGGREGATION (per Item ID)

monthly = df.groupby(['Item ID', 'Year_Month']).agg(
    Item_Description   = ('Item Description', 'first'),
    Category           = ('Category', 'first'),
    Total_Qty_Sold     = ('Quantity', 'sum'),
    Num_Orders         = ('Order ID', 'nunique'),
    Num_Days_With_Sales= ('Date', 'nunique'),
    Avg_Unit_Price     = ('Unit Price When Sold', 'mean'),
    Median_Unit_Price  = ('Unit Price When Sold', 'median'),
    Min_Unit_Price     = ('Unit Price When Sold', 'min'),
    Max_Unit_Price     = ('Unit Price When Sold', 'max'),
    Std_Unit_Price     = ('Unit Price When Sold', 'std'),
    Total_Revenue      = ('Revenue', 'sum'),
    Avg_Qty_Per_Order  = ('Quantity', 'mean'),
).reset_index()

# Fill NaN std
monthly['Std_Unit_Price'] = monthly['Std_Unit_Price'].fillna(0)

# Round numeric columns
for col in ['Avg_Unit_Price', 'Median_Unit_Price', 'Std_Unit_Price',
            'Total_Revenue', 'Avg_Qty_Per_Order']:
    monthly[col] = monthly[col].round(2)

# Convert period back to usable date columns
monthly['Year'] = monthly['Year_Month'].dt.year
monthly['Month'] = monthly['Year_Month'].dt.month
monthly['Year_Month_Str'] = monthly['Year_Month'].astype(str)

# LAG FEATURES (previous months' demand)
monthly = monthly.sort_values(['Item ID', 'Year_Month']).reset_index(drop=True)

for lag in [1, 2, 3]:
    monthly[f'Qty_Lag_{lag}'] = (
        monthly.groupby('Item ID')['Total_Qty_Sold']
        .shift(lag)
    )

# Rolling averages
monthly['Qty_Rolling_3M_Avg'] = (
    monthly.groupby('Item ID')['Total_Qty_Sold']
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
).round(2)

monthly['Qty_Rolling_6M_Avg'] = (
    monthly.groupby('Item ID')['Total_Qty_Sold']
    .transform(lambda x: x.shift(1).rolling(6, min_periods=1).mean())
).round(2)

# Price change from previous month
monthly['Prev_Avg_Price'] = monthly.groupby('Item ID')['Avg_Unit_Price'].shift(1)
monthly['Price_Change_Pct'] = (
    ((monthly['Avg_Unit_Price'] - monthly['Prev_Avg_Price']) / monthly['Prev_Avg_Price'] * 100)
    .round(2)
)

# Month-over-month quantity growth
monthly['Prev_Qty'] = monthly.groupby('Item ID')['Total_Qty_Sold'].shift(1)
monthly['Qty_Growth_Pct'] = (
    ((monthly['Total_Qty_Sold'] - monthly['Prev_Qty']) / monthly['Prev_Qty'] * 100)
    .round(2)
)

# PRODUCT-LEVEL STATIC FEATURES
product_stats = df.groupby('Item ID').agg(
    Lifetime_Total_Qty   = ('Quantity', 'sum'),
    Lifetime_Avg_Price   = ('Unit Price When Sold', 'mean'),
    First_Sale_Date      = ('Date', 'min'),
    Last_Sale_Date       = ('Date', 'max'),
).reset_index()

product_stats['Lifetime_Avg_Price'] = product_stats['Lifetime_Avg_Price'].round(2)

monthly = monthly.merge(product_stats, on='Item ID', how='left')

# Months since first sale
monthly['Period_Start'] = monthly['Year_Month'].apply(lambda p: p.start_time)
monthly['Months_Since_Launch'] = (
    (monthly['Period_Start'] - monthly['First_Sale_Date']).dt.days / 30.44
).round(0).astype(int)


# CLEAN UP & EXPORT
# Fill NaN in lag/rolling/growth columns with 0
fill_cols = ['Qty_Lag_1', 'Qty_Lag_2', 'Qty_Lag_3',
             'Qty_Rolling_3M_Avg', 'Qty_Rolling_6M_Avg',
             'Price_Change_Pct', 'Qty_Growth_Pct']
monthly[fill_cols] = monthly[fill_cols].fillna(0)

drop_cols = ['Year_Month', 'Prev_Avg_Price', 'Prev_Qty', 'Period_Start',
             'First_Sale_Date', 'Last_Sale_Date']
monthly = monthly.drop(columns=drop_cols)

# Reorder columns
col_order = [
    'Item ID', 'Item_Description', 'Category',
    'Year_Month_Str', 'Year', 'Month',
    # Target
    'Total_Qty_Sold',
    # Order features
    'Num_Orders', 'Num_Days_With_Sales', 'Avg_Qty_Per_Order',
    # Price features
    'Avg_Unit_Price', 'Median_Unit_Price', 'Min_Unit_Price',
    'Max_Unit_Price', 'Std_Unit_Price', 'Price_Change_Pct',
    # Revenue
    'Total_Revenue',
    # Lag features
    'Qty_Lag_1', 'Qty_Lag_2', 'Qty_Lag_3',
    'Qty_Rolling_3M_Avg', 'Qty_Rolling_6M_Avg',
    'Qty_Growth_Pct',
    # Product-level features
    'Lifetime_Total_Qty', 'Lifetime_Avg_Price', 'Months_Since_Launch',
]
monthly = monthly[col_order]

# Save
monthly.to_csv('../data/processed/Monthly_Demand_Dataset.csv', index=False)
