
import pandas as pd
from src.data.datasets.plant_traits_dataset import PlantTraitsDataset
from src.data.helpers.dataset_scaler import Dataset_Scaler
from sklearn.preprocessing import StandardScaler


def _get_dataset():
    train_path = "data/processed/planttraits2024/train.feather"
    train_df = pd.read_feather(train_path)
    dataset = PlantTraitsDataset(train_df, stage="train")    
    return dataset

def test_dataset_scaler():
    scaler = Dataset_Scaler(scaler_type=StandardScaler)
    dataset = _get_dataset()
    original_shape = dataset.data.shape

    scaler.scale_dataset(dataset)
    assert dataset.data is not None
    assert type(dataset.data) == pd.DataFrame
    assert dataset.data.shape == original_shape


if __name__ == "__main__":
    test_dataset_scaler()
    print("Dataset scaler tests passed")
