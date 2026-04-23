import pandas as pd
from pathlib import Path

# Setup paths
data_root = Path('../data/raw')
all_chunks = []

print("Starting data consolidation...")

# Iterate through each year folder
for year_folder in sorted(data_root.iterdir()):
    if year_folder.is_dir() and year_folder.name.isdigit():
        print(f"--- Processing Year: {year_folder.name} ---")
        
        # Find all "Details" files
        details_files = list(year_folder.glob("Orders - * - Order Details.csv"))
        
        for d_file in details_files:
            # Identify the corresponding "Items" file
            i_file = year_folder / d_file.name.replace("Order Details", "Order Items")
            
            if i_file.exists():
                
                df_details = pd.read_csv(d_file, header=1)
                df_items = pd.read_csv(i_file, header=1)
                
                # Clean column names
                df_details.columns = [str(c).strip() for c in df_details.columns]
                df_items.columns = [str(c).strip() for c in df_items.columns]

                # If 'Order ID' isn't found, try header=0 as a fallback
                if 'Order ID' not in df_details.columns:
                    df_details = pd.read_csv(d_file, header=0)
                    df_items = pd.read_csv(i_file, header=0)
                    df_details.columns = [str(c).strip() for c in df_details.columns]
                    df_items.columns = [str(c).strip() for c in df_items.columns]

                # Drop empty columns
                df_details = df_details.dropna(axis=1, how='all')
                df_items = df_items.dropna(axis=1, how='all')

                try:
                    # Extract Date and Status from Details into Items
                    merged = pd.merge(
                        df_items, 
                        df_details[['Order ID', 'Date Order Made', 'Status']], 
                        on='Order ID', 
                        how='inner'
                    )
                    
                    # Filter for RESOLVED
                    resolved = merged[merged['Status'].str.upper() == 'RESOLVED'].copy()
                    
                    # Select required columns
                    final_chunk = resolved[[
                        'Order ID', 'Date Order Made', 'Item ID', 
                        'Item Description', 'Quantity', 'Unit Price When Sold'
                    ]]
                    
                    all_chunks.append(final_chunk)
                    print(f"  Successfully merged: {d_file.name}")
                    
                except KeyError as e:
                    print(f"  Error in {d_file.name}: Column {e} not found.")
                    print(f"  Available columns: {df_details.columns.tolist()}")
            else:
                print(f"  Warning: Missing Items file for {d_file.name}")

# Combine everything into one Master DataFrame
if all_chunks:
    master_df = pd.concat(all_chunks, ignore_index=True)
    master_df['Order ID'] = master_df['Order ID'].astype(str)
    
    # Save to your local directory
    output_path = '../data/processed/Processed_Data.csv'
    master_df.to_csv(output_path, index=False)
    
    print("\n" + "="*30)
    print(f"PROCESS COMPLETE!")
    print(f"Total Rows: {len(master_df)}")
else:
    print("\nNo data processed. Check folder paths and file names.")
