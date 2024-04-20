
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import torch 
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50
from torchvision.models.resnet import import ResNet50_Weights

from src.data.helpers.transform_holder import TransformHolder

"""
Precompute ResNet or ConvNeXt embeddings for images
"""

class SimpleImageDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)
                            if file.endswith(('.jpeg'))]
        self.ids = [os.path.basename(f).split(".")[0] for f in self.image_files]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        id = self.ids[idx]
        img_name = self.image_files[idx]
        image = Image.open(img_name).convert('RGB')

        if self.transform:
            image = self.transform(image=np.array(image))

        return image, id


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data
    data_dir = "data/raw/planttraits2024/train_images"
    transform = TransformHolder.get_train_transform()

    dataset = SimpleImageDataset(data_dir, transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)

    # Set up model
    model = resnet50(pretrained=True, weights=ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity() # Remove the final classification layer to get embeddings
    model = model.to(device)
    model.eval()

    # Compute Embeddings
    embeddings = []
    ids = []
    with torch.no_grad():
        for inputs, ids_here in data_loader:
            inputs = inputs['image'].to(device)
            output = model(inputs)
            embeddings.append(torch.tensor(output.cpu().numpy()))
            ids.extend(list(ids_here))

    # Build df of embeddings with filenames
    embeddings = torch.cat(embeddings)
    embeddings_df = pd.DataFrame(embeddings.numpy())
    embeddings_df["id"] = ids

    # Save embeddings
    embeddings_path = "data/processed/planttraits2024/embeddings.feather"
    embeddings_df.to_feather(embeddings_path)

    print(f"Embeddings computed saved to {embeddings_path}.")



if __name__ == "__main__":
    main()
