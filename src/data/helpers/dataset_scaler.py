
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

    # Scale the numeric columns of df
    def scale_df(self, df: pd.DataFrame):
        # Get the numeric columns
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        num_df = df[numeric_cols]

        scaled_array = self.scaler.fit_transform(num_df)
        num_df_scaled = pd.DataFrame(scaled_array, columns=num_df.columns)

        # Replace the numeric columns with the scaled columns
        df[numeric_cols] = num_df_scaled
        return df
