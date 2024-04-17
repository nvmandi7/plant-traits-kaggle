
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_Scaler:

    def __init__(self, scaler_type=StandardScaler):
        self.scaler = scaler_type()
        pass
    
    def scale_dataset(self, dataset: Dataset):
        df = self.scale_df(dataset.data)
        dataset.data = df

        # Potentially add wandb logging here later

    def scale_df(self, df: pd.DataFrame):
        scaled_array = self.scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_array, columns=df.columns)
        return df_scaled

