from pathlib import Path
import pandas as pd
from pprint import pprint

import json














# Replace this with your target directory
folder_path = Path('xxxx/generalizable_safety/datasets_box')

# Find all files that end with 'prompt_train.parquet'
matching_files = list(folder_path.rglob('*prompt_train.parquet'))

sampled_dfs = []
# Cold start 1500 samples
sample_num = 80
for file in matching_files:
    #print(file)
    df = pd.read_parquet(file)
    # Randomly sample 80 rows
    sample_size = min(sample_num, len(df))
    sampled_df = df.sample(n=sample_size, random_state=42)  # Set random_state for reproducibility
    sampled_dfs.append(sampled_df)

# Step 3: Combine all sampled DataFrames
combined_df = pd.concat(sampled_dfs, ignore_index=True)
# Step 4: Convert to JSON
json_output = combined_df.to_json(orient='records', lines=True)
# Optional: Save to a file
#with open('sampled_combined.json', 'w') as f:
#    f.write(json_output)
