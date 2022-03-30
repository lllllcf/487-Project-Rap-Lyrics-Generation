import numpy as np
import torch
import random
from datetime import datetime
from torch.nn.functional import softmax


def generate_rap(generator, sen_input, num_sentences, max_words, dataset):
    random.seed(datetime.now())
    res = []
    n = len(sen_input.split(' '))
    for sen in range(num_sentences):
        num_words = random.randint(4, max_words + 1)
        word_list = []
        # fixed input or update input
        for word in range(num_words):
            word_list.append( predict(dataset, generator, sen_input) )
        sen_input = " ".join(word_list[-4:])
        if sen == 0:
            res.append(" ".join(word_list[0:]))
        else:
            res.append(" ".join(word_list[n:]))
    return res


# can be change into fixed input size and update state
def predict(dataset, model, input_text):
    words = input_text.split(' ')
    model.eval()

    state_h, state_c = model.init_state(len(words))

    x = torch.tensor([[dataset.word_to_index[w] for w in words]])
    y_pred, (state_h, state_c) = model(x, (state_h, state_c))

    last_word_logits = y_pred[0][-1]
    p = softmax(last_word_logits, dim=0).detach().numpy()
    word_index = np.random.choice(len(last_word_logits), p=p)
    words = dataset.index_to_word[word_index]

    return words
