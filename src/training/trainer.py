import logging

LOGGER = logging.getLogger(__name__)


# Define Trainer configuration
trainer_config = {
    'max_epochs': 15,               # Specify the maximum number of epochs
    'logger': wandb_logger,         # Use WandB logger
    'precision': '16-mixed',        # Use mixed precision (16-bit)
#     'accumulate_grad_batches': 2,   # Accumulate gradients over multiple batches (if needed)
    'check_val_every_n_epoch': 1,   # Validate every epoch
    'callbacks': [
        # Add any additional callbacks if needed
        pl.callbacks.LearningRateMonitor(logging_interval='step'),  # Log learning rate
        pl.callbacks.ModelCheckpoint(dirpath='./models/',  monitor="val_r2", mode="max", save_top_k=1),
        pl.callbacks.ModelCheckpoint(dirpath='./models/',  monitor="val_loss", mode="min", save_top_k=1)
    ],
    'benchmark': True,
}


class Trainer:
    """Trainer is responsible for model training."""

    def __init__(
        self,
        model,
        train_dataloader,
        val_dataloader,
        optimizer,
        config,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.epochs = config.epochs
        self.device = config.device

    def run(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)

    def train_epoch(self, epoch):
        self.model.train()

        for _batch_index, (input, targets) in enumerate(self.train_dataloader):
            input = input.to(self.device)
            targets = targets.to(self.device)

            output = self.model(input)
            loss = self.criterion(output, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def validate_epoch(self, epoch):
        self.model.eval()
