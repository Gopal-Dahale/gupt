""" EMNIST Data Module Class"""
import os
import json
import numpy as np
import h5py
import torch
from torchvision import transforms
from torchvision.datasets import EMNIST
from torch.utils.data import random_split
from gupt.data.base_data_module import BaseDataModule, load_data
from gupt.data.base_dataset import BaseDataset

# Directory to hold downloaded dataset
DATA_DIR = os.getcwd().replace('\\', '/') + '/datasets/downloaded'
TRAINING_DATA_FRACTION = 0.9
PROCESSED_DATA_PATH = os.getcwd().replace('\\',
                                          '/') + '/datasets/processed/EMNIST'
PROCESSED_DATA_FILENAME = PROCESSED_DATA_PATH + "/emnist_byclass.h5"
EMNIST_MAPPING_FILE_PATH = os.getcwd().replace(
    '\\', '/') + '/gupt/data/emnist_mapping.json'

# Special tokens used for detecting lines and paragraphs
# - Blank token
# - Start token
# - End token
# - Padding token
SPECIAL_TOKENS = ["<B>", "<S>", "<E>", "<P>"]
COUNT_SPECIAL_TOKENS = len(SPECIAL_TOKENS)


class EMNISTDataModule(BaseDataModule):
    """Data Module for EMNIST Dataset

    Args:
        BaseDataModule (Module): Base Data Module Class
    """
    def __init__(self, args=None):
        super().__init__(args)
        self.data_dir = DATA_DIR
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.dims = (1, 28, 28)
        self.output_dims = (1, )
        if not os.path.exists(EMNIST_MAPPING_FILE_PATH):
            download_and_process_emnist(self.data_dir)
        with open(EMNIST_MAPPING_FILE_PATH, 'r') as file:
            emnist_mapping = json.load(file)
        self.mapping = emnist_mapping['mapping']
        self.inverse_mapping = {
            char: idx
            for idx, char in enumerate(self.mapping)
        }

    def prepare_data(self):
        """Download train/ test data
        """
        if not os.path.exists(PROCESSED_DATA_FILENAME):
            download_and_process_emnist(self.data_dir)
        with open(EMNIST_MAPPING_FILE_PATH, 'r') as file:
            emnist_mapping = json.load(file)
            self.mapping = emnist_mapping['mapping']

    def setup(self, stage=None):
        """Split the data into training, validation and test set.

        Args:
            stage (str, optional): used to separate setup logic for trainer.{fit,validate,test}. If setup is called with stage = None, we assume all stages have been set-up. Defaults to None.
        """
        # Assign Train/val split(s) for use in Dataloaders
        x_train = []
        y_train = []
        x_test = []
        y_test = []

        with h5py.File(PROCESSED_DATA_FILENAME, 'r') as file:
            x_train = file['x_train'][:]
            y_train = torch.LongTensor(file['y_train'][:])
            x_test = file['x_test'][:]
            y_test = torch.LongTensor(file['y_test'][:])

        emnist_full = BaseDataset(x_train, y_train, self.transform)
        emnist_test = BaseDataset(x_test, y_test, self.transform)

        train_data_size = int(len(emnist_full) * TRAINING_DATA_FRACTION)
        val_data_size = len(emnist_full) - train_data_size

        self.train_data, self.val_data = random_split(
            emnist_full, [train_data_size, val_data_size])

        # Assign Test split(s) for use in Dataloaders
        self.test_data = emnist_test


def download_and_process_emnist(data_dir):
    """Download and process emnist dataset

    Args:
        data_dir (str): Path where the downloaded emnist dataset will be stored
    """
    train_data = EMNIST(data_dir, split='byclass', train=True,
                        download=True)  # Training Data
    test_data = EMNIST(data_dir, split='byclass', train=False,
                       download=True)  # Test Data

    # Process the train and test data
    # We need to perform swap axes because the images are rotated in a wrong orientation

    x_train = train_data.data.reshape(-1, 28, 28).swapaxes(1, 2)
    y_train = train_data.targets + COUNT_SPECIAL_TOKENS

    x_test = test_data.data.reshape(-1, 28, 28).swapaxes(1, 2)
    y_test = test_data.targets + COUNT_SPECIAL_TOKENS

    # Balance dataset
    x_train, y_train = balance_dataset(x_train, y_train)
    x_test, y_test = balance_dataset(x_test, y_test)

    # Store data in hdf5 format
    os.makedirs(PROCESSED_DATA_PATH)

    with h5py.File(PROCESSED_DATA_FILENAME, 'w') as file:

        # Training dataset
        file.create_dataset("x_train",
                            data=x_train,
                            dtype="u1",
                            compression="gzip")
        file.create_dataset("y_train",
                            data=y_train,
                            dtype="u1",
                            compression="gzip")

        # Test dataset
        file.create_dataset("x_test",
                            data=x_test,
                            dtype="u1",
                            compression="gzip")
        file.create_dataset("y_test",
                            data=y_test,
                            dtype="u1",
                            compression="gzip")

    # Store EMNIST character mapping
    mapping = emnist_character_mapping(train_data.classes)
    emnist_mapping = {"mapping": mapping}
    with open(EMNIST_MAPPING_FILE_PATH, "w") as file:
        json.dump(emnist_mapping, file)


def balance_dataset(images, labels):
    """Balances dataset by taking at most the mean number of samples per class

    Args:
        images (torch.Tensor): Images (Inputs)
        labels (torch.Tensor): Labels (Outputs)
    """

    # Count number of occurrences of each value in labels
    # and then find the mean value
    samples_per_class = int(np.bincount(labels).mean())
    unique_labels = np.unique(labels)
    new_indices = []
    for label in unique_labels:

        indices = np.where(labels == label)[
            0]  # Indices at which 'label' is present in 'labels'
        sampled_indices = list(
            np.unique(np.random.choice(indices, size=samples_per_class))
        )  # Take at most mean number of unique indices

        new_indices.extend(sampled_indices)

    sampled_labels = labels[new_indices]
    sampled_images = images[new_indices]

    return sampled_images, sampled_labels


def emnist_character_mapping(classes):
    """Create mapping for EMNIST dataset

    Args:
        classes (list): Existing list of unique labels of EMNIST

    Returns:
        mapping (list): character mapping for EMNIST dataset
    """
    symbols = [
        " ",
        "!",
        '"',
        "#",
        "&",
        "'",
        "(",
        ")",
        "*",
        "+",
        ",",
        "-",
        ".",
        "/",
        ":",
        ";",
        "?",
    ]
    mapping = SPECIAL_TOKENS + classes + symbols
    return mapping


if __name__ == "__main__":
    load_data(EMNISTDataModule)
