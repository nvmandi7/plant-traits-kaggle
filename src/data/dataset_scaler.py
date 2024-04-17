
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_Scaler:

    def __init__(self, scaler_type=StandardScaler):
        self.scaler = scaler_type()
        pass
    
    def scale_dataset(self, dataset: Dataset):
        df = self._scale_df(dataset.data)
        dataset.data = df

        # Potentially add wandb logging here later

    def _scale_df(self, df: pd.DataFrame):
        df_scaled = self.scaler.fit_transform(df)
        # df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
        return df_scaled

