import json

import numpy as np

from collections import Counter

import matplotlib.pyplot as plt


path_vocab1_freq = 'stat/vocab1_freq.json'
path_len1_freq = 'stat/len1_freq.json'
path_vocab2_freq = 'stat/vocab2_freq.json'
path_len2_freq = 'stat/len2_freq.json'

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial Unicode MS']


def count(path_freq, items, field):
    pairs = Counter(items)
    sort_items = [item for item, freq in pairs.most_common()]
    sort_freqs = [freq for item, freq in pairs.most_common()]
    item_freq = dict()
    for item, freq in zip(sort_items, sort_freqs):
        item_freq[item] = freq
    with open(path_freq, 'w') as f:
        json.dump(item_freq, f, ensure_ascii=False, indent=4)
    plot_freq(sort_items, sort_freqs, field, u_bound=20)


def plot_freq(items, freqs, field, u_bound):
    inds = np.arange(len(items))
    plt.bar(inds[:u_bound], freqs[:u_bound], width=0.5)
    plt.xlabel(field)
    plt.ylabel('freq')
    plt.xticks(inds[:u_bound], items[:u_bound], rotation='vertical')
    plt.show()


def statistic(path_train):
    with open(path_train, 'r') as f:
        pairs = json.load(f)
    text1s, text2s = zip(*pairs)
    text1s, text2s = list(text1s), list(text2s)
    all_word1s = ' '.join(text1s).split()
    all_word2s = ' '.join(text2s).split()
    text1_lens = [len(text.split()) for text in text1s]
    text2_lens = [len(text.split()) for text in text2s]
    count(path_vocab1_freq, all_word1s, 'vocab')
    count(path_len1_freq, text1_lens, 'text_len')
    count(path_vocab2_freq, all_word2s, 'vocab')
    count(path_len2_freq, text2_lens, 'text_len')


if __name__ == '__main__':
    path_train = 'data/train.json'
    statistic(path_train)
