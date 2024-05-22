
import pandas as pd
import numpy as np

from src.data.helpers.dataset_scaler import Dataset_Scaler

"""
Scale dataset and convert to feather file. All downstream tasks should use the feather file instead of csv.
"""

def main():
    train_path = "data/raw/planttraits2024/train.csv"
    trait_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']

    # All trails except X4 have distributions skewed by outliers
    trait_columns = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    drop_cols = ['X4_sd', 'X11_sd', 'X18_sd', 'X26_sd', 'X50_sd', 'X3112_sd']
    scaler = Dataset_Scaler(exclude_cols=[])

    # Read csv with id column as string
    train_df = pd.read_csv(train_path, dtype={"id": str})
    train_df = train_df.drop(columns=drop_cols, axis=1)
    print("Train data shape: ", train_df.shape)

    # Apply log transformation to traits
    train_df[trait_columns] = train_df[trait_columns].apply(np.log1p)

    # Scale dataset
    train_df = scaler.scale_df(train_df)

    # Generate species identifier
    train_df['species'] = train_df.groupby(trait_columns).ngroup()
    species_counts = train_df['species'].nunique()
    print (f"{species_counts} unique species found in {len(train_df)} records")

    train_df.to_feather("data/processed/planttraits2024/train_log_trans_zscore_targets.feather")
    return



if __name__ == "__main__":
    main()
