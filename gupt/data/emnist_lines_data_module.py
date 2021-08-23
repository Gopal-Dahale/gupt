""" EMNISTLines Data Module Class"""
import os
import h5py
import numpy as np
import torch
from gupt.data.base_data_module import BaseDataModule, load_data
from gupt.data.emnist_data_module import EMNISTDataModule
from gupt.data.sentence_builder import SentenceBuilder
from gupt.data.base_dataset import BaseDataset

# Directory to hold downloaded dataset
DATA_DIR = BaseDataModule.dataset_dir() / 'processed/EMNIST_LINES'
print(DATA_DIR)


class EMNISTLinesDataModule(BaseDataModule):
    """Data Module for EMNIST lines Dataset

    Args:
        BaseDataModule (Module): Base Data Module Class
    """

    def __init__(self, args=None):
        super().__init__(args)
        self.min_overlap = 0  # Minimum overlap between two images
        self.max_overlap = 0.3  # Maximum overlap between two images
        self.train_size = 15000  # Size of training set
        self.val_size = 5000  # Size of validation set
        self.test_size = 4000  # Size of test set
        self.limit = 30  # Maximum length of a line
        self.allow_start_end_tokens = True  # Add start and end tokens at the start and end of line respectively

        self.emnist = EMNISTDataModule()
        self.mapping = self.emnist.mapping

        emnist_dims = self.emnist.dims
        self.dims = (
            emnist_dims[0],  # Number of channels
            emnist_dims[1],  # Height of line
            emnist_dims[2] * self.limit  # Length ofl line
        )
        self.output_dims = (self.limit, 1)

        # Setup EMNIST
        self.emnist.prepare_data()
        self.emnist.setup()

    def prepare_data(self):
        """Prepare train/ test data
        """
        if not os.path.exists(self.file_name()):
            try:
                os.makedirs(DATA_DIR)
            except OSError as error:
                print("Directory Already exists", error)

            for split in ["train", "val", "test"]:
                self._synthesize_data(split)

    def setup(self, stage=None):

        # Assign Train/val/test split(s) for use in Dataloaders
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        x_test = []
        y_test = []
        with h5py.File(self.file_name(), 'r') as file:
            x_train = file['x_train'][:]
            y_train = file['y_train'][:]
            x_val = file['x_val'][:]
            y_val = file['y_val'][:]
            x_test = file['x_test'][:]
            y_test = file['y_test'][:]

        emnist_lines_train = BaseDataset(x_train, y_train)
        emnist_lines_val = BaseDataset(x_val, y_val)
        emnist_lines_test = BaseDataset(x_test, y_test)

        self.train_data = emnist_lines_train
        self.val_data = emnist_lines_val
        self.test_data = emnist_lines_test

    def file_name(self):
        """Name of the file in which EMNISTLines dataset is stored

        Returns:
            (str): File name
        """
        name = f"train_{self.train_size}_val_{self.val_size}_test_{self.test_size}_mino_{self.min_overlap}_maxo_{self.max_overlap}_lim_{self.limit}_s_e_tok_{self.allow_start_end_tokens}"
        return DATA_DIR / (name + ".h5")

    def _synthesize_data(self, split):
        """Create synthetic EMNISTLines dataset using EMNIST dataset

        Args:
            split (str): Type of split (train, val, test)
        """
        print("EMNISTLinesDataModule is synthesizing data for " + split)
        size = 0
        character_table = {}
        if split == 'train':
            size = self.train_size
            character_table = generate_character_table(self.emnist.train_data,
                                                       self.mapping)
        elif split == 'val':
            size = self.val_size
            character_table = generate_character_table(self.emnist.val_data,
                                                       self.mapping)
        elif split == 'test':
            size = self.test_size
            character_table = generate_character_table(self.emnist.test_data,
                                                       self.mapping)

        with h5py.File(self.file_name(), 'a') as file:
            images, labels = generate_emnist_lines(character_table, self.limit,
                                                   self.min_overlap,
                                                   self.max_overlap, size,
                                                   self.dims)

            labels = string_to_label(labels, self.allow_start_end_tokens,
                                     self.emnist.inverse_mapping,
                                     self.output_dims[0])

            file.create_dataset("x_" + split,
                                data=images,
                                dtype="float32",
                                compression="gzip")
            file.create_dataset("y_" + split,
                                data=labels,
                                dtype="u1",
                                compression="gzip")


def string_to_label(labels, allow_start_end_tokens, inv_mapping, max_length):
    """Converts list of strings to list of labels using inverse mapping

    Args:
        labels (list): List of strings
        allow_start_end_tokens (bool): Specifies whether to append start and end token
        inv_mapping (dict): Mapping from indices to characters
        max_length (int): Maximum length of a string

    Returns:
        res (numpy.ndarray): 2d array of labels where each row is the inverse mapping of particular sentence
    """
    res = inv_mapping["<P>"] * np.ones((len(labels), max_length))
    for i, label in enumerate(labels):
        chars = list(label)
        if allow_start_end_tokens:
            chars = ["<S>"] + chars + ["<E>"]
        for j, char in enumerate(chars):
            res[i, j] = inv_mapping[char]
    return res


def generate_character_table(data, mapping):
    """Create character table which is a dictionary with EMNIST classes as keys and their corresponding images as values.
    Each value is a list of images for each key.

    Args:
        data (gupt.data.base_dataset.BaseDataset): Dataset (train/test/val)
        mapping (list): List of EMNIST character mapping

    Returns:
        character_table (dict): Character table
    """
    character_table = {}
    for image, label in data:
        char = mapping[label]
        if char not in character_table:
            character_table[char] = []
        character_table[char].append(image)
    return character_table


def generate_emnist_lines(character_table, limit, min_overlap, max_overlap,
                          size, dims):
    """Create EMNISTLines dataset using EMNIST

    Args:
        character_table (dict): A dictionary with EMNIST classes as keys and their corresponding images as values.
        Each value is a list of images for each key.
        limit (int): Maximum length of string
        min_overlap (float): Minimum overlap of two images
        max_overlap (float): Maximum overlap of two images
        size (int): size of the dataset i.e. number of images
        dims (tuple): Dimension on input

    Returns:
        (tuple): Images and their correponding labels
    """
    s_builder = SentenceBuilder()
    s_length = limit - 2  # Two is subtracted for start and end token

    images = []
    labels = []
    count = size
    while len(images) != count:
        sentences = s_builder.build(limit=s_length, count=(count - len(images)))
        labels.extend(sentences)
        for sentence in sentences:
            images.append(
                string_to_image(character_table, min_overlap, max_overlap,
                                sentence, dims))

    images = torch.stack(images)
    return images, labels


def string_to_image(character_table, min_overlap, max_overlap, sentence, dims):
    """Convert given string to image

    Args:
        character_table (dict): A dictionary with EMNIST classes as keys and their corresponding images as values.
        Each value is a list of images for each key.
        min_overlap (float): Minimum overlap of two images
        max_overlap (float): Maximum overlap of two images
        sentence (str): Sentence for which image is to be constructed
        dims ([type]): [description]

    Returns:
        line (torch.Tenosr): Corresponding image of given sentence
    """
    overlap = np.random.uniform(low=min_overlap, high=max_overlap)

    empty_image = torch.zeros((28, 28), dtype=torch.float32)

    # Images of sentence's characters
    sentence_character_table = {}
    for letter in sentence:
        if letter not in sentence_character_table:
            char_img = empty_image
            choices = character_table.get(letter, [])
            if choices:
                idx = np.random.choice(len(choices))
                char_img = choices[idx]
            sentence_character_table[letter] = char_img.reshape(28, 28)

    sentence_images = [sentence_character_table[letter] for letter in sentence]

    img_height, img_width = sentence_images[0].shape
    concatenated_img_width = img_width - int(overlap * img_width)
    line_width = dims[-1]
    line = torch.zeros((img_height, line_width), dtype=torch.float32)

    break_point = 0
    for img in sentence_images:
        line[:, break_point:break_point + img_width] += img
        break_point += concatenated_img_width

    return line


if __name__ == "__main__":
    load_data(EMNISTLinesDataModule)
