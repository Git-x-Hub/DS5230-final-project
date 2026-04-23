# E-Commerce Demand Forecasting

Comparing ARIMA, Gradient Boosting, and DeepAR for monthly product demand 
prediction on a Philippine e-commerce dataset (773 products, 37 months).

## Data
Raw order data is organized by year in the following structure:

```
data/
└── raw/
    ├── 2023/
    ├── 2024/
    ├── 2025/
    └── 2026/
```

Each year folder contains paired CSV files:
- `Orders - * - Order Details.csv` (order date, status)
- `Orders - * - Order Items.csv` (product, quantity, price)

Step 1 (`data_preprocessing.py`) merges these files, filters resolved orders, and outputs a single `data/processed/Processed_Data.csv`.


## Reproducibility
Baseline models use `random_state=42` for deterministic results.
DeepAR involves stochastic training (weight initialization, mini-batch shuffling); 
results may vary slightly between runs.


## Setup
For PyTorch with GPU (CUDA), install separately first:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Then install all dependencies:
```bash
pip install -r requirements.txt
```


## Pipeline
```bash

cd src

# Step 1: Preprocess raw transaction data
python data_preprocessing.py

# Step 2: Inspect unique product descriptions (optional)
python util.py

# Step 3: Categorize products
python categorize.py

# Step 4: Aggregate to monthly dataset with engineered features
python aggregate_demand.py

# Step 5: Exploratory data analysis
python eda_analysis.py

# Step 6: Train ML baseline models (Ridge, RF, GB, HistGB)
python baseline_ml.py

# Step 7: Train ARIMA + SMA baseline and compare with GB
python baseline_arima_sma.py

# Step 8: Train DeepAR and generate 3-way comparison
python advanced_deepar.py

## Notes
- Run all steps in sequence (Step 1 through Step 8). Each step depends on the output of the previous step.
- Step 5 requires LaTeX for plot rendering — set `text.usetex: True` in `eda_analysis.py` if available
- DeepAR runs on CPU (~30–40 min) or GPU (~10 min)
