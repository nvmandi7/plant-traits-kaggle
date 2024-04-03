import logging

LOGGER = logging.getLogger(__name__)


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
