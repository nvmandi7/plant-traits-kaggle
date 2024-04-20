
import pandas as pd

from src.data.helpers.dataset_scaler import Dataset_Scaler

"""
Scale dataset and convert to feather file. All downstream tasks should use the feather file instead of csv.
"""

def main():
    scaler = Dataset_Scaler()
    train_path = "data/raw/planttraits2024/train.csv"
    drop_cols = ['X4_sd', 'X11_sd', 'X18_sd', 'X26_sd', 'X50_sd', 'X3112_sd']

    # Read csv with id column as string
    train_df = pd.read_csv(train_path, dtype={"id": str})
    train_df = train_df.drop(columns=drop_cols, axis=1)
    train_df = scaler.scale_df(train_df)
    print("Train data shape: ", train_df.shape)

    train_df.to_feather("data/processed/planttraits2024/train.feather")
    return



if __name__ == "__main__":
    main()