
import pandas as pd

from src.data.preprocessing.precompute_embeddings import main, SimpleImageDataset   


def display_embeddings():
    embeddings_path = "tests/fixtures/data/processed/planttraits2024/train_embeddings.feather"
    embeddings_df = pd.read_feather(embeddings_path)
    print(embeddings_df)
    return


def test_main():
    data_dir = "tests/fixtures/data/raw/planttraits2024/train_images"
    model_name = "resnet50"
    embeddings_path = "tests/fixtures/data/processed/planttraits2024/train_embeddings.feather"
    main(data_dir, model_name, embeddings_path)

    display_embeddings()
    return

