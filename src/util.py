import pandas as pd

# Load your data
df = pd.read_csv('../data/processed/Processed_Data.csv')

# Get unique descriptions and sort them
unique_items = sorted(df['Item Description'].unique().tolist())

# Print them as a list you can copy
for item in unique_items:
    print(item)
