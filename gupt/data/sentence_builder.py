""" Sentence Builder class"""
import os
import re
import itertools
import string
import numpy as np
import nltk
from gupt.data.base_data_module import BaseDataModule

# Directory to hold downloaded dataset
DATA_DIR = BaseDataModule.dataset_dir() / 'downloaded/NLTK'


class SentenceBuilder:
    """Sentence Builder class
    """

    def __init__(self):
        self.corpus = corpus_string()
        self.word_indices = [0] + [
            i.start() + 1 for i in re.finditer(" ", self.corpus)
        ]  # Indices at which all words start

    def build(self, limit, count=5):
        """Build 'count' number of strings from corpus text of length at most 'limit'

        Args:
            limit (int): Maximum length of string
            count (int, optional): Number of strings to generate. Defaults to 5.

        Returns:
            sentences (list): List of sentences
        """
        sentences = []  # List of all sentences
        attempts = 20  # Number of attempts to make before returning from the function

        while count and attempts:

            # Generate a random index from the word_indices list
            random_idx = np.random.randint(low=0,
                                           high=len(self.word_indices) - 1)

            start_idx = self.word_indices[random_idx]  # Get the index of word

            # Holds the possible candidates for end indices of a sentence constraint to length of string i.e. limit
            end_idxes = []

            for i in range(random_idx + 1, len(self.word_indices)):
                if (self.word_indices[i] - start_idx > limit) or (count == 0):
                    break
                end_idxes.append(self.word_indices[i])
                count -= 1

            # Extract sentences from corpus
            for end_idx in end_idxes:
                sentences.append(self.corpus[start_idx:end_idx].strip())
            attempts -= 1

        return sentences


def corpus_string():
    """String containing brown corpus text will all punctuations removed"""

    nltk.data.path.append(DATA_DIR)  # Add DATA_DIR to nltk data path

    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        nltk.download('brown', download_dir=DATA_DIR)

    corpus = nltk.corpus.brown.sents()

    # concat all lists with " "
    corpus = " ".join(itertools.chain.from_iterable(corpus))

    punctuations = string.punctuation  # string of all punctuations

    # Translation table, which is a mapping of Unicode ordinals to None (here,
    # but can also be to Unicode ordinals, strings).
    translation_table = {ord(p): None for p in punctuations}
    corpus = corpus.translate(translation_table)  # Remove punctutations
    corpus = ' '.join(corpus.split())  # Remove contiguous spaces

    return corpus
