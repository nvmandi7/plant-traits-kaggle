
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

"""
Lightning model that takes in precomputed image embeddings, tabular data, and runs a small MLP
"""
class BaselineModel(L.LightningModule):
    def __init__(self, image_model_name, table_num_features, intermediate_num_features, num_targets, learning_rate=1e-3):
        super(BaselineModel, self).__init__()

        # Image model
        self.image_model = timm.create_model(image_model_name, pretrained=True, num_classes=0)
        self.image_model_to_intermediate = torch.nn.Linear(self.image_model.num_features, intermediate_num_features)

        # Define the table model
        self.table_model = torch.nn.Sequential(
            torch.nn.Linear(table_num_features, 512),
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.Dropout(0.25),
            nn.ReLU(),
            torch.nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Batch normalization layer
            nn.Dropout(0.25),
            nn.ReLU(),
            torch.nn.Linear(256, intermediate_num_features)
        )

        # Fully connected layers with dropout and batch normalization
        self.fc_combined = nn.Sequential(
            nn.Linear(intermediate_num_features, 512),
            nn.BatchNorm1d(512),  # Batch normalization layer
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),  # Batch normalization layer
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(256, num_targets),
        )

        self.learning_rate = learning_rate

        # Metric for tracking R2 score
        self.r2_score = R2Score(num_outputs=6)

    def forward(self, x_image, x_table):
        x_image = self.image_model(x_image)
        x_image = self.image_model_to_intermediate(x_image)
        x_table = self.table_model(x_table)
        x_combined = (x_image + x_table)/2
        output = self.fc_combined(x_combined)
        return output
    
    def get_image_intermediate(self, x_image, x_table):
        x_image = self.image_model(x_image)
        x_image = self.image_model_to_intermediate(x_image)
        return x_image
    
    def get_table_intermediate(self, x_image, x_table):
        x_table = self.table_model(x_table)
        return x_table

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate,
                    steps_per_epoch=len(train_dataloader),
                    epochs=self.trainer.max_epochs
                ),
                'interval': 'step',
            },
        }

    def training_step(self, batch, batch_idx):
        x_image, x_table, y_true = batch
        y_pred = self(x_image, x_table)
        loss = F.mse_loss(y_pred, y_true)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_r2(y_pred, y_true, 'train_r2')
        return loss

    def validation_step(self, batch, batch_idx):
        x_image, x_table, y_true = batch
        y_pred = self(x_image, x_table)
        loss = F.mse_loss(y_pred, y_true)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log_r2(y_pred, y_true, 'val_r2')
        return loss

    def log_r2(self, y_pred, y_true, name):
        r2 = self.r2_score(y_pred, y_true)
        self.log(name, r2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def configure_wandb_logger(self, model_name):
        # Get the current date and time in the specified format
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Generate a random 4-letter ID
        random_id = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz', k=4))
        # Combine date, time, and random ID to form the run name
        run_name = f"{current_datetime}_{random_id}"
        # Initialize the WandB logger with project name and run name
        wandb_logger = pl.loggers.wandb.WandbLogger(project="PlantTraits2024", name=run_name, log_model=True)
        # Log the model name
        wandb_logger.log_hyperparams({"model_name": model_name})
        return wandb_logger
