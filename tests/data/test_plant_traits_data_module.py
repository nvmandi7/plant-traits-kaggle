
import os
import pandas as pd
from src.config.training_config import TrainingConfig
from src.data.plant_traits_data_module import PlantTraitsDataModule




if __name__ == "__main__":
    # Temp workaround
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


    config = TrainingConfig()
    data_module = PlantTraitsDataModule(config)

    print("Data Module: ", data_module)
    print("Data dir: ", data_module.data_dir)
    print("Batch size: ", data_module.batch_size)
    print("Transform: ", data_module.transform)
