
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset


class Dataset_Scaler:

    def __init__(self, scaler_type=StandardScaler, exclude_cols=[]):
        self.scaler = scaler_type()
        self.exclude_cols = exclude_cols
        return
    
    def scale_dataset(self, dataset: Dataset):
        df = self.scale_df(dataset.data)
        dataset.data = df
        # Potentially add wandb logging here later
        return

    # Scale the numeric columns of df, excluding the exclude_cols
    def scale_df(self, df: pd.DataFrame):
        # Drop the exclude_cols
        temp_df = df.drop(columns=self.exclude_cols, axis=1, inplace=False)

        # Get the numeric columns
        numeric_cols = temp_df.select_dtypes(include=['float64', 'int64']).columns
        num_df = temp_df[numeric_cols]

        scaled_array = self.scaler.fit_transform(num_df)
        num_df_scaled = pd.DataFrame(scaled_array, columns=num_df.columns)

        # Replace the numeric columns with the scaled columns
        df[numeric_cols] = num_df_scaled
        return df
