import random
from datetime import datetime
import string
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import copy


def get_dev_data(data, percentage):
    n = int(len(data) * percentage / 2)
    return data[n:-n], data[:n] + data[-n:]


def merge_lists(lists):
    res = []
    for list in lists:
        for sen in list:
            res.append(sen)
    return res


def add_some_news(dis_rap_raw, news_data):
    # rap : news = 1 : 1
    random.seed(datetime.now())
    new_news = merge_lists(news_data)

    n = len(dis_rap_raw)
    res = random.sample(new_news, n)

    return res


def dis_pre_data_preprocession(rap_news):
    # true rap 1, fake rap 0
    n = int(len(rap_news) / 2)
    res = []
    for i in range(n * 2):
        if i < n:
            res.append({'X': rap_news[i], 'y': 1})
        else:
            res.append({'X': rap_news[i], 'y': 0})
    return res


def add_some_music(rap, music, percent):
    random.seed(datetime.now())
    new_rap = merge_lists(rap)
    new_music = merge_lists(music)

    num_music_sen = int(round(len(new_rap) * percent))
    music_sen = random.sample(new_music, num_music_sen)
    for x in music_sen:
        new_rap.insert(random.randint(0, len(new_rap)), x)

    return new_rap


def gen_pre_data_preprocession(rap_music, percent, gen_clean_control):
    random.seed(datetime.now())
    n = int(percent * len(rap_music))
    i_start = random.randint(0, int(len(rap_music) - n - 1))
    gen_pre = rap_music[i_start: (i_start + n)]
    dis_rap = rap_music[0:i_start] + rap_music[(i_start + n):]

    gen_pre = gen_clean(gen_pre, gen_clean_control)

    return rap_music, gen_pre, dis_rap


def gen_clean(text, control):
    text = text.copy()

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    for i in range(len(text)):
        text[i] = extract_word(text[i])
        if control['lemmatize']:
            for j in range(len(text[i])):
                text[i][j] = lemmatizer.lemmatize(text[i][j])

    if control['stop_words']:
        for i in range(len(text)):
            for j in range(len(text[i])):
                if text[i][j] in stop_words:
                    text[i][j] = ""

    if control['remove_number']:
        for i in range(len(text)):
            for j in range(len(text[i])):
                if text[i][j].isnumeric():
                    text[i][j] = ""

    for i in range(len(text)):
        text[i] = ' '.join(text[i])

    return (' '.join(text)).split(' ')


def extract_word(input_s):
    input_string = input_s
    pu = string.punctuation
    for p in pu:
        input_string = input_string.replace(p, ' ')
    return input_string.lower().split()


def plot_loss(stats):
    """Plot generator loss and discriminator loss."""
    plt.plot(stats['loss_ind'], stats['dis_loss'], label='Discriminator loss')
    plt.plot(stats['loss_ind'], stats['gen_loss'], label='Generator loss')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.show()
