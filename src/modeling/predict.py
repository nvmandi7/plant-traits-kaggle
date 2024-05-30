
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import torch 
from torch.utils.data import Dataset, DataLoader

from torchvision.models import convnext_base
from torchvision.models.convnext import ConvNeXt_Base_Weights


from src.data.datasets.plant_traits_dataset import PlantTraitsDataset
from src.modeling.plant_traits_model import PlantTraitsModel
from src.data.helpers.transform_holder import TransformHolder

"""
Script to generate predictions on test.csv for PTDataset and PTModel
"""

def main(exponentiate_targets, model_file, outfile):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data
    transform = TransformHolder.get_val_transform()
    data_df = pd.read_csv("data/raw/planttraits2024/test.csv", dtype={"id": str})
    dataset = PlantTraitsDataset(data_df, transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=1)

    # Set up model
    model = PlantTraitsModel.load_from_checkpoint(model_file)
    model = model.to(device)
    model.eval()

    # Compute Predictions
    # embeddings = []
    # ids = []
    # with torch.no_grad():
    #     for inputs, ids_here in data_loader:
    #         inputs = inputs['image'].to(device)
    #         output = model(inputs)
    #         embeddings.append(torch.tensor(output.cpu().numpy()))
    #         ids.extend(list(ids_here))

    # # Build df of embeddings with filenames
    # embeddings = torch.cat(embeddings)
    # embeddings_df = pd.DataFrame(embeddings.numpy())
    # embeddings_df["id"] = ids
    # embeddings_df.columns = embeddings_df.columns.astype(str)

    # # Save embeddings
    # embeddings_df.to_feather(embeddings_path)
    # print(f"Embeddings computed saved to {embeddings_path}.")



if __name__ == "__main__":
    exponentiate_targets = True # Set to True if targets are log transformed
    model_file = "models/epoch=473-step=1422.ckpt"
    outfile = f"data/outputs/predictions_{model_file}.csv"

    main(exponentiate_targets, model_file, outfile)
