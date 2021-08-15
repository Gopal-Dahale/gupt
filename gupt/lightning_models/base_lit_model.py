import pytorch_lightning as pl
import torch
from torchmetrics import Accuracy
import torch.nn.functional as F


class BaseLitModel(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.args = {}
        if args is not None:
            self.args = vars(args)
        self.model = model
        self.lr = 1e-3
        self.loss = F.cross_entropy
        self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr,momentum=0.9)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()
        self.test_acc = Accuracy()

    def forward(self,x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        