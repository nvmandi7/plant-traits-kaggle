
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torchmetrics import R2Score

"""
Lightning model that takes in precomputed image embeddings, tabular data, and runs a small MLP to regress plant traits

By default it expect and input size of 2048 resnet embedding dims + 163 tabular features = 2211
"""

class BaselineModel(L.LightningModule):
    def __init__(self, input_dims=2211, learning_rate=1e-3):
        super(BaselineModel, self).__init__()

        # Fully connected layers with dropout and batch normalization
        def fc_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.Dropout(0.25),
                nn.ReLU(),
            )

        self.mlp = nn.Sequential(
            fc_block(input_dims, 512),
            fc_block(512, 128),
            fc_block(128, 64),
            nn.Linear(64, 6)
        )
        
        self.learning_rate = learning_rate
        self.best_val_loss = float('inf')

        # Metric for tracking R2 score
        self.r2_score = R2Score(num_outputs=6)



    def forward(self, x):
        output = self.mlp(x)
        return output
    
    # ---------------------

    def _shared_step(self, batch, batch_idx):
        row, targets = batch
        preds = self(row)
        loss = F.mse_loss(preds, targets)
        return loss, preds, targets

    def _log_r2(self, metric_name, preds, targets):
        r2 = self.r2_score(preds, targets)
        self.log(metric_name, r2, on_step=False, on_epoch=True, prog_bar=True, logger=True)
    
    
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self._log_r2('train_r2', preds, targets)
        return {'loss': loss, 'preds': preds, 'targets': targets}

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self._log_r2('val_r2', preds, targets)
        return {'loss': loss, 'preds': preds, 'targets': targets}
    


    def train_epoch_end(self, outputs):
        # Average training loss and R2 across all batches
        average_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('average_train_loss', average_loss)
        print(f"Average training loss for epoch: {average_loss}")


    def on_validation_epoch_end(self, outputs):
        # Average validation loss across all batches
        average_loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('average_val_loss', average_loss)
        print(f"Average validation loss for epoch: {average_loss}")

        if average_loss < self.best_val_loss:
            self.best_val_loss = average_loss
            self.log('best_val_loss', self.best_val_loss)
            print("New best val loss.")

    # ---------------------


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
        
