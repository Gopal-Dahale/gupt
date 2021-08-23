""" Base Data Module Class"""
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader

DATA_DIR = Path(__file__).resolve().parents[2] / "datasets"


class BaseDataModule(pl.LightningDataModule):
    """Base Data Module Class. All other data classes will inherit from this class

    Args:
        pl (Module): Lightning Data Module
    """

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
        self.val_data = None
        self.test_data = None

    def config(self):
        """Function to return configuration of dataset

        Returns:
            dict: Input and Output dimensions of dataset and output mapping
        """
        return {
            'input_dims': self.dims,
            'output_dims': self.output_dims,
            'mapping': self.mapping
        }

    @classmethod
    def dataset_dir(cls):
        """Dataset directory path

        Returns:
            (str): Dataset directory path
        """
        return DATA_DIR

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers)


def load_data(data_module):
    """Function to load data from given data module

    Args:
        data_module (class): Data module class (eg. MNIST)
    """
    dataset = data_module()
    dataset.prepare_data()
    dataset.setup()
    print("Dataset Loaded Successfully")
