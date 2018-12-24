import pickle as pk

import numpy as np

import torch
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from preprocess import clean

from represent import sent2ind

from nn_arch import PtrEncode, PtrDecode, PtrPlot

from util import map_item


plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['Arial Unicode MS']


def load_model(name, embed_mat, device, mode):
    embed_mat = torch.Tensor(embed_mat)
    model = torch.load(map_item(name, paths), map_location=device)
    full_dict = model.state_dict()
    arch = map_item('_'.join([name, mode]), archs)
    part = arch(embed_mat).to(device)
    part_dict = part.state_dict()
    part_dict = {key: val for key, val in full_dict.items() if key in part_dict}
    part_dict.update(part_dict)
    part.load_state_dict(part_dict)
    return part


def load_plot(name, embed_mat, device):
    embed_mat = torch.Tensor(embed_mat)
    model = torch.load(map_item(name, paths), map_location=device)
    full_dict = model.state_dict()
    arch = map_item(name + '_plot', archs)
    plot = arch(embed_mat).to(device)
    plot_dict = plot.state_dict()
    plot_dict = {key: val for key, val in full_dict.items() if key in plot_dict}
    plot_dict.update(plot_dict)
    plot.load_state_dict(plot_dict)
    return plot


def ind2word(word_inds):
    ind_words = dict()
    for word, ind in word_inds.items():
        ind_words[ind] = word
    return ind_words


def check(probs, cand, keep_eos):
    max_probs, max_inds = list(), list()
    sort_probs = -np.sort(-probs)
    sort_inds = np.argsort(-probs)
    for prob, ind in zip(list(sort_probs), list(sort_inds)):
        if not keep_eos and ind == eos_ind:
            continue
        if ind not in skip_inds:
            max_probs.append(prob)
            max_inds.append(ind)
        if len(max_probs) == cand:
            break
    return max_probs, max_inds


def search(decode, state, cand):
    pad_bos = sent2ind([bos], word_inds, seq_len, 'post', keep_oov=True)
    word2 = torch.LongTensor([pad_bos]).to(device)
    probs = decode(word2, state)[0][0]
    probs = F.softmax(probs, dim=0).numpy()
    max_probs, max_inds = check(probs, cand, keep_eos=False)
    text2s, log_sums = [bos] * cand, np.log(max_probs)
    fin_text2s, fin_logs = list(), list()
    next_words, count = [ind_words[ind] for ind in max_inds], 1
    while cand > 0:
        log_mat, ind_mat = list(), list()
        count = count + 1
        for i in range(cand):
            text2s[i] = ' '.join([text2s[i], next_words[i]])
            pad_seq2 = sent2ind(text2s[i], word_inds, seq_len, 'post', keep_oov=True)
            sent2 = torch.LongTensor([pad_seq2]).to(device)
            step = min(count - 1, seq_len - 1)
            probs = decode(sent2, state)[0][step]
            probs = F.softmax(probs, dim=0).numpy()
            max_probs, max_inds = check(probs, cand, keep_eos=True)
            max_logs = np.log(max_probs) + log_sums[i]
            log_mat.append(max_logs)
            ind_mat.append(max_inds)
        max_logs = -np.sort(-np.array(log_mat), axis=None)[:cand]
        next_text2s, next_words, log_sums = list(), list(), list()
        for log in max_logs:
            args = np.where(log_mat == log)
            sent_arg, ind_arg = int(args[0][0]), int(args[1][0])
            next_word = ind_words[ind_mat[sent_arg][ind_arg]]
            if next_word != eos and count < max_len:
                next_words.append(next_word)
                next_text2s.append(text2s[sent_arg])
                log_sums.append(log)
            else:
                cand = cand - 1
                fin_text2s.append(text2s[sent_arg])
                fin_logs.append(log / count)
        text2s = next_text2s
    max_arg = np.argmax(np.array(fin_logs))
    return fin_text2s[max_arg][1:]


device = torch.device('cpu')

seq_len = 50
max_len = 50

bos, eos = '*', '#'

pad_ind, oov_ind = 0, 1

path_embed = 'feat/embed.pkl'
path_word_ind = 'feat/word_ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)

skip_inds = [pad_ind, oov_ind]

eos_ind = word_inds[eos]

ind_words = ind2word(word_inds)

archs = {'ptr_encode': PtrEncode,
         'ptr_decode': PtrDecode,
         'ptr_plot': PtrPlot}

paths = {'ptr': 'model/rnn_ptr.pkl'}

models = {'ptr_encode': load_model('ptr', embed_mat, device, 'encode'),
          'ptr_decode': load_model('ptr', embed_mat, device, 'decode'),
          'ptr_plot': load_plot('ptr', embed_mat, device)}


def plot_ptr(word1s, word2s, atts):
    len1, len2 = len(word1s), len(word2s)
    atts = atts[:len2, -len1:]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cax = ax.matshow(atts.numpy(), cmap='bone')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + word1s, rotation='vertical')
    ax.set_yticklabels([''] + word2s)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()


def predict(text, name):
    text1 = clean(text)
    text1 = ' '.join([text1, eos])
    word1s = text1.split()
    pad_seq1 = sent2ind(word1s, word_inds, seq_len, 'pre', keep_oov=True)
    sent1 = torch.LongTensor([pad_seq1]).to(device)
    encode = map_item(name + '_encode', models)
    decode = map_item(name + '_decode', models)
    with torch.no_grad():
        encode.eval()
        state = encode(sent1)
        decode.eval()
        pred = search(decode, state, cand=3)
        if __name__ == '__main__':
            text2 = ' '.join([bos, pred])
            word2s = text2.split()
            pad_seq2 = sent2ind(word2s, word_inds, seq_len, 'post', keep_oov=True)
            sent2 = torch.LongTensor([pad_seq2]).to(device)
            plot = map_item(name + '_plot', models)
            atts = plot(sent1, sent2)[0]
            plot_ptr(word1s[:-1], word2s[1:] + [eos], atts)
        return pred


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('ptr: %s' % predict(text, 'ptr'))
