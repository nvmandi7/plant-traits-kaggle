
import os
import pandas as pd
import torch 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
from torch.utils.data import DataLoader

from src.data.helpers.transform_holder import TransformHolder

"""
Precompute ResNet or ConvNeXt embeddings for images
"""

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up data
    data_dir = "data/raw/planttraits2024/train_images"
    transform = TransformHolder.get_train_transform()

    dataset = ImageFolder(root=data_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Set up model
    model = resnet50(pretrained=True)
    model.fc = torch.nn.Identity() # Remove the final classification layer to get embeddings
    model = model.to(device)
    model.eval()

    # Compute Embeddings
    embeddings = []
    filenames = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            output = model(inputs)
            embeddings.append(output.cpu().numpy())
            filenames.extend([dataset.imgs[i][0] for i in labels.tolist()])

    # Build df of embeddings with filenames
    embeddings = torch.cat(embeddings)
    embeddings_df = pd.DataFrame(embeddings.numpy())
    embeddings_df["id"] = [os.path.basename(f).split(".")[0] for f in filenames]
    embeddings_df = embeddings_df.set_index("id")

    # Save embeddings
    embeddings_path = "data/processed/planttraits2024/embeddings.feather"
    

    print(f"Embeddings computed saved to {embeddings_path}.")



if __name__ == "__main__":
    main()
