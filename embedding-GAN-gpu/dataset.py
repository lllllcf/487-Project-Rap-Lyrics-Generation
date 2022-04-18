from ast import arg
import torch
from collections import Counter
from torch.utils.data import Dataset
from sentence_transformers import SentenceTransformer
import numpy as np
from train_utils import dis_sen2vec
import nltk
nltk.download('punkt')

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
    def __init__(self, args, data, all_data, embed):
        self.args = args
        self.words = data
        self.dict = all_data
        self.uniq_words = self.get_uniq_words()
        # self.embed = embed
        # self.unk = unk # np.mean([self.embed[word] for word in self.embed.index_to_key], axis=0)

        # self.index_to_word = {index: word for index, word in enumerate(self.uniq_words)}
        self.word_to_index = {word: index for index, word in enumerate(self.uniq_words)}
        # self.word_to_vec = self.get_wv()
        self.words_indexes = [self.word_to_index[w] for w in self.words]

    # def get_wv(self):
    #     res = {}
    #     for _, word in enumerate(self.uniq_words):
    #         if word in self.embed:
    #             res[word] = self.embed[word]
    #         else:
    #             res[word] = self.unk
    #     return res

    def get_uniq_words(self):
        word_counts = Counter(self.dict)
        return sorted(word_counts, key=word_counts.get, reverse=True)

    def __len__(self):
        return len(self.words_indexes) - self.args["sequence_length"]

    def __getitem__(self, index):
        return (
            self.words[index:index+self.args["sequence_length"]],
            self.words[index+1:index+self.args["sequence_length"] + 1],
        )
    
    # def get_x(self, index):
    #     if self.words[index] in self.embed:
    #         res = self.embed[self.words[index]].reshape(1, 300)
    #     else:
    #         res = self.unk.reshape(1, 300)
    #     for i in range(index + 1, index+self.args["sequence_length"]):
    #         if self.words[i] in self.embed:
    #             res = np.concatenate((res, self.embed[self.words[i]].reshape(1, 300)), axis=0)
    #         else:
    #             res = np.concatenate((res, self.unk.reshape(1, 300)), axis=0)
    #     return res

    def get_sen(self, input):
        res = self.index_to_word[input[0, 0].item()]
        for i in range(len(input[0])):
            if i != 0:
                res += ' '
                res += self.index_to_word[input[0, i].item()]
        return res


class DISDataset(Dataset):
    """Dataset for modified QA task on SQuAD2.0"""

    def __init__(self, data, embed, tfidf):
        self.embed = embed
        self.tfidf = tfidf
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        d = self.data[idx]
        return (
            torch.tensor(dis_sen2vec([d['X']], self.embed, self.tfidf)).cuda(),
            torch.tensor(d['y']).cuda(),
        )