
import os
import pickle as pkl
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader

from torchvision.models import convnext_base
from torchvision.models.convnext import ConvNeXt_Base_Weights


from src.data.helpers.dataset_scaler import Dataset_Scaler
from src.data.datasets.plant_traits_dataset import PlantTraitsDataset
from src.modeling.plant_traits_model import PlantTraitsModel
from src.data.helpers.transform_holder import TransformHolder

"""
Script to generate predictions on test.csv for PTDataset and PTModel

Preprocess: just scale according to train.csv mu and sigma
Postprocess: exponentiate if necessary, and unscale
"""

def get_scaler():
    path = "data/processed/latest_scaler.pkl"
    scaler = Dataset_Scaler(exclude_cols=['id'])
    scaler.load_from_pkl(path)
    return scaler

def main(exponentiate_targets, model_file, outfile):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Preprocess dataset - scale according to train.csv mu and sigma
    scaler = get_scaler()
    data_df = pd.read_csv("data/raw/planttraits2024/test.csv", dtype={"id": int})
    data_df = scaler.scale_df(data_df, fit=False)

    # Set up dataset/loader
    transform = TransformHolder.get_val_transform()
    dataset = PlantTraitsDataset(data_df, transform=transform, stage="test")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    # Set up model
    model = PlantTraitsModel.load_from_checkpoint(model_file, learning_rate=0.001)
    model = model.to(device)
    model.eval()

    # Compute Predictions
    predictions = []
    with torch.no_grad():
        for image, row, _ in data_loader:
            image = image.to(device)
            row = row.to(device)

            preds = model(image, row)
            predictions.append(preds.cpu())

    # Make DataFrame
    predictions = torch.cat(predictions).numpy()
    columns = dataset.target_cols
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df.to_csv(f"data/outputs/predictions_unprocessed.csv")

    # Postprocess predictions - exponentiate if necessary, and unscale
    if exponentiate_targets:
        predictions_df = np.exp(predictions_df)
    predictions_df = scaler.unscale_df(predictions_df)

    # Save predictions
    predictions_df.to_csv(outfile, index=False)


if __name__ == "__main__":
    exponentiate_targets = True # Set to True if targets are log transformed
    model_file = "models/self.config.experiment_name=0-epoch=24-val_r2=0.37.ckpt"
    outfile = f"data/outputs/predictions_{model_file}.csv"

    main(exponentiate_targets, model_file, outfile)
