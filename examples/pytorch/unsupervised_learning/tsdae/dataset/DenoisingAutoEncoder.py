import os
from pathlib import Path

from torch.utils.data import Dataset
from typing import Union, List
import numpy as np
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

DATA_DIR = "data"

class InputExample:
    """
    Structure for one input example with texts, the label and a unique id
    """
    def __init__(self, guid: str = '', texts: List[str] = None,  label: Union[int, float] = 0):
        """
        Creates one InputExample with the given texts, guid and label


        :param guid
            id for the example
        :param texts
            the texts for the example.
        :param label
            the label for the example
        """
        self.guid = guid
        self.texts = texts
        self.label = label

    def __str__(self):
        return "<InputExample> label: {}, texts: {}".format(str(self.label), "; ".join(self.texts))


class DenoisingAutoEncoderDataset(Dataset):
    """
    The DenoisingAutoEncoderDataset returns InputExamples in the format: texts=[noise_fn(sentence), sentence]
    It is used in combination with the DenoisingAutoEncoderLoss: Here, a decoder tries to re-construct the
    sentence without noise.

    :param sentences: A list of sentences
    :param noise_fn: A noise function: Given a string, it returns a string with noise, e.g. deleted words
    """
    def __init__(self, sentences: List[str], noise_fn=lambda s: DenoisingAutoEncoderDataset.delete(s)):
        self.sentences = sentences
        self.noise_fn = noise_fn


    def __getitem__(self, item):
        sent = self.sentences[item]
        return InputExample(texts=[self.noise_fn(sent), sent])


    def __len__(self):
        return len(self.sentences)

    # Deletion noise.
    @staticmethod
    def delete(text, del_ratio=0.6):
        words = nltk.word_tokenize(text)
        n = len(words)
        if n == 0:
            return text

        keep_or_not = np.random.rand(n) > del_ratio
        if sum(keep_or_not) == 0:
            keep_or_not[np.random.choice(n)] = True # guarantee that at least one word remains
        words_processed = TreebankWordDetokenizer().detokenize(np.array(words)[keep_or_not])
        return words_processed


def read_corpus_for_pretrain(filename):
    train_sentences = []
    data_path = Path(os.path.join(DATA_DIR, filename))
    if data_path.is_file():
        with open(data_path, 'rt', encoding='utf8') as fIn:
            for line in fIn:
                train_sentences.append(line.strip())
    else:
        raise FileNotFoundError(f"Please put {filename} in data folder for training TSDAE.")
    return train_sentences
