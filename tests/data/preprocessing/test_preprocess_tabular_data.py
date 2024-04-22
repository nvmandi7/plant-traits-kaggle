
import pandas as pd

from src.data.helpers.dataset_scaler import Dataset_Scaler
from src.data.preprocessing.preprocess_tabular_data import main


def test_dtypes():
    scaler = Dataset_Scaler()
    train_path = "data/raw/planttraits2024/train.csv"

    # Read csv with id column as string
    train_df = pd.read_csv(train_path, dtype={"id": str})
    print("Number of NaNs: ", train_df.isnull().sum().sum())
    assert train_df["id"].dtype == "object"

    train_df = scaler.scale_df(train_df)
    print("Train data shape: ", train_df.shape)
    assert train_df["id"].dtype == "object"
    print("Number of NaNs: ", train_df.isnull().sum().sum())

    return

def test_main():
    main()

    # Assure id column is str
    train_path = "data/processed/planttraits2024/train.feather"
    train_df = pd.read_feather(train_path)
    assert train_df["id"].dtype == "object"

    # No NaNs
    assert train_df.isnull().sum().sum() == 0

    # Assure species column is created
    assert "species" in train_df.columns

    return