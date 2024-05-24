
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L

from torchmetrics import R2Score

"""
Lightning model that takes in plant images and tabular data.
It trains a vision encoder on the plant image, a small MLP on tabular data, and then concatenates the two to regress plant traits
"""

class PlantTraitsModel(L.LightningModule):
    def __init__(self, input_image_dims, output_image_dims=256, input_tabular_dims=163, output_tabular_dims=32, output_final_dims=16, learning_rate=1e-3, scheduler_args=None):
        super(PlantTraitsModel, self).__init__()

        # Fully connected layers with dropout and batch normalization
        def fc_block(in_dim, out_dim):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                # nn.BatchNorm1d(out_dim),
                nn.Dropout(0.25),
                nn.ReLU(),
            )

        self.mlp = nn.Sequential(
            fc_block(input_dims, 512),
            # fc_block(512, 256),
            fc_block(512, 128),
            fc_block(128, 64),
            nn.Linear(64, output_dims)
        )
        
        self.learning_rate = learning_rate
        # self.scheduler_args = scheduler_args
        self.best_val_loss = float('inf')
        self.r2_score = R2Score(num_outputs=output_dims)

        self.training_step_outputs = []
        self.validation_step_outputs = []


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
        r2 = self.r2_score(preds, targets) # Averages r2 across all 6 plant traits
        self.log(metric_name, r2, on_step=True, on_epoch=True, prog_bar=True)
    
    def _log_random_plant_traits(self, preds, targets):
        random_plant_idx = torch.randint(0, len(preds), (1,))
        random_plant_pred = preds[random_plant_idx].squeeze()
        random_plant_target = targets[random_plant_idx].squeeze()
        col_names = ['X4', 'X11', 'X18', 'X26', 'X50', 'X3112']
        for i in range(len(random_plant_pred)):
            self.log(f"pred_{col_names[i]}", random_plant_pred[i], on_step=True, on_epoch=False)
            self.log(f"target_{col_names[i]}", random_plant_target[i], on_step=True, on_epoch=False)
    
    def training_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self._log_r2('train_r2', preds, targets)
        # self._log_random_plant_traits(preds, targets)
        
        outputs = {'loss': loss, 'preds': preds, 'targets': targets}
        self.training_step_outputs.append(outputs)
        return outputs

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self._shared_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        self._log_r2('val_r2', preds, targets)

        outputs = {'loss': loss, 'preds': preds, 'targets': targets}
        self.validation_step_outputs.append(outputs)
        return outputs


    def on_train_epoch_end(self):
        # Average training loss and R2 across all batches
        average_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
        self.log('average_train_loss', average_loss)
        print(f"Average training loss for epoch: {average_loss}")


    def on_validation_epoch_end(self):
        # Average validation loss across all batches
        average_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        self.log('average_val_loss', average_loss)
        print(f"Average validation loss for epoch: {average_loss}")

        if average_loss < self.best_val_loss:
            self.best_val_loss = average_loss
            self.log('best_val_loss', self.best_val_loss)
            print("New best val loss.")
        
    # ---------------------
            
    def configure_optimizers(self):
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return self.optimizer
        self.optimizer_config = {
            'optimizer': self.optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.OneCycleLR(
                    self.optimizer,
                    max_lr=self.learning_rate,
                    steps_per_epoch=self.scheduler_args['steps_per_epoch'],
                    epochs=self.scheduler_args['epochs']
                ),
                'interval': 'step',
            },
        }
        return self.optimizer_config