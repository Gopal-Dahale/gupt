""" Base Lightning Model"""
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics


class Accuracy(torchmetrics.Accuracy):
    """Measuring Accuracy

    Args:
        torchmetrics (Module): Accuracy metric from torchmetrics
    """

    def update(self, preds, target):
        """Applies softmax function if the preds are not between 0 and 1

        Args:
            preds (tensor): Model Predictions
            target (tensor): Ground Truth
        """
        if preds.min() < 0 or preds.max() > 1:
            preds = torch.nn.functional.softmax(preds, dim=-1)
        super().update(preds=preds, target=target)


class BaseLitModel(pl.LightningModule):
    """Base Lightning Model. All models inherits this class.

    Args:
        pl (module): Lightning Module
    """

    def __init__(self, model, args):
        super().__init__()
        self.args = {}
        if args is not None:
            self.args = vars(args)

        self.model = model
        self.lr = 1e-3
        self.loss_func = F.cross_entropy
        self.optimizer = None

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer, max_lr=self.lr, total_steps=100)

        return {
            'optimizer': self.optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

    def training_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.train_acc(y_hat, y)
        self.log("train_loss", loss)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        y_hat = self.model(x)
        loss = self.loss_func(y_hat, y)
        self.val_acc(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc",
                 self.val_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)

    def test_step(self, batch, batch_idx):  # pylint: disable=unused-argument
        x, y = batch
        y_hat = self.model(x)
        self.test_acc(y_hat, y)
        self.log("test_acc",
                 self.test_acc,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=True)
