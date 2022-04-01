from ast import arg
import torch
from collections import Counter
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer

sen_embed = SentenceTransformer('bert-base-nli-mean-tokens')

def basic_collate_fn(batch):
    X = []
    y = []
    for (i, j) in batch:
        X.append(i)
        y.append(j)

    return torch.cat(X), y


class GENDataset(Dataset):
    # data is list of words
    def __init__(self, args, data, all_data):
        self.args = args
        self.words = data
        self.dict = all_data
        self.uniq_words = self.get_uniq_words()

        self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}

        self.words_indexes = [self.word_to_index[w] for w in self.words]

    def get_uniq_words(self):
        word_counts = Counter(self.dict)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args["sequence_length"]

    def __getitem__(self, index):
        return (
            torch.tensor(self.words_indexes[index:index+self.args["sequence_length"]]),
            torch.tensor(self.words_indexes[index+1:index+self.args["sequence_length"] + 1]),
        )

    def get_sen(self, input):
        res = self.index_to_word[input[0, 0].item()]
        for i in range(len(input[0])):
            if i != 0:
                res += ' '
                res += self.index_to_word[input[0, i].item()]
        return res


class DISDataset(Dataset):
    """Dataset for modified QA task on SQuAD2.0"""

    def __init__(self, data, sen_embed):
        self.sen_embed = sen_embed
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.tensor(self.sen_embed.encode([d['X']])),
            torch.tensor(d['y']),
        )
