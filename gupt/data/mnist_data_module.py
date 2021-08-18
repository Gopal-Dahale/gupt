""" MNIST Data Module Class"""
import os
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from gupt.data.base_data_module import BaseDataModule, load_data

# Directory to hold downloaded dataset
DATA_DIR = os.getcwd().replace('\\', '/') + '/datasets/downloaded'


class MNISTDataModule(BaseDataModule):
    """Data Module for MNIST Dataset

    Args:
        BaseDataModule (Module): Base Data Module Class
    """

    def __init__(self, args):
        super().__init__(args)
        self.data_dir = DATA_DIR
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.1307), (0.3081))])
        self.dims = (1, 28, 28)
        self.output_dims = (1,)
        self.mapping = list(range(10))

    def prepare_data(self):
        """Download train/ test data
        """
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        """Split the data into training, validation and test set.

        Args:
            stage (str, optional): used to separate setup logic for trainer.{fit,validate,test}. If setup is called with stage = None, we assume all stages have been set-up. Defaults to None.
        """

        # Assign Train/val split(s) for use in Dataloaders
        mnist_full = MNIST(self.data_dir,
                           train=True,
                           download=True,
                           transform=self.transform)
        self.train_data, self.val_data = random_split(mnist_full, [55000, 5000])

        # Assign Test split(s) for use in Dataloaders
        self.test_data = MNIST(self.data_dir,
                               train=False,
                               download=True,
                               transform=self.transform)


if __name__ == "__main__":
    load_data(MNISTDataModule)
