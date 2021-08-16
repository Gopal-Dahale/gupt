import pytorch_lightning as pl
from torch.utils.data import DataLoader

class BaseDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = {}
        if args is not None:
            self.args = vars(args)
        self.batch_size = 128
        self.num_workers = 4

        self.dims = None
        self.output_dims = None
        self.mapping = None
        self.train_data = None
        self.val_data= None
        self.test_data = None

    def config(self):
        return {
            'input_dims':self.dims,
            'output_dims':self.output_dims,
            'mapping':self.mapping
        }

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size,shuffle=True,num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers)

def load_data(data_module):
    dataset = data_module(None)
    dataset.prepare_data()
    dataset.setup()
    print("Dataset Loaded Successfully")