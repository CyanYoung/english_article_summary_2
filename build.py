import time

import pickle as pk

import torch
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from nn_arch import Ptr

from util import map_item


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detail = False if torch.cuda.is_available() else True

batch_size = 32

path_embed = 'feat/embed.pkl'
path_word_ind = 'feat/word_ind.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)

archs = {'ptr': Ptr}

paths = {'ptr': 'model/rnn_ptr.pkl'}


def load_feat(path_feats):
    with open(path_feats['sent1_train'], 'rb') as f:
        train_sent1s = pk.load(f)
    with open(path_feats['sent2_train'], 'rb') as f:
        train_sent2s = pk.load(f)
    with open(path_feats['label_train'], 'rb') as f:
        train_labels = pk.load(f)
    with open(path_feats['sent1_dev'], 'rb') as f:
        dev_sent1s = pk.load(f)
    with open(path_feats['sent2_dev'], 'rb') as f:
        dev_sent2s = pk.load(f)
    with open(path_feats['label_dev'], 'rb') as f:
        dev_labels = pk.load(f)
    return train_sent1s, train_sent2s, train_labels, dev_sent1s, dev_sent2s, dev_labels


def step_print(step, batch_loss, batch_acc):
    print('\n{} {} - loss: {:.3f} - acc: {:.3f}'.format('step', step, batch_loss, batch_acc))


def epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra):
    print('\n{} {} - {:.2f}s - loss: {:.3f} - acc: {:.3f} - val_loss: {:.3f} - val_acc: {:.3f}'.format(
          'epoch', epoch, delta, train_loss, train_acc, dev_loss, dev_acc) + extra)


def tensorize(feats, device):
    tensors = list()
    for feat in feats:
        tensors.append(torch.LongTensor(feat).to(device))
    return tensors


def get_loader(triples):
    sent1s, sent2s, labels = triples
    triples = TensorDataset(sent1s, sent2s, labels)
    return DataLoader(triples, batch_size, shuffle=True)


def get_metric(model, loss_func, triples):
    sent1s, sent2s, labels = triples
    labels = labels.view(-1)
    num = (labels > 0).sum().item()
    probs = model(sent1s, sent2s)
    probs = probs.view(-1, probs.size(-1))
    logs = torch.log(probs)
    preds = torch.max(logs, dim=1)[1]
    loss = loss_func(logs, labels)
    acc = (preds == labels).sum().item()
    return loss, acc, num


def batch_train(model, loss_func, optim, loader, detail):
    total_loss, total_acc, total_num = [0] * 3
    for step, triples in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(model, loss_func, triples)
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
        if detail:
            step_print(step + 1, batch_loss / batch_num, batch_acc / batch_num)
    return total_loss / total_num, total_acc / total_num


def batch_dev(model, loss_func, loader):
    total_loss, total_acc, total_num = [0] * 3
    for step, triples in enumerate(loader):
        batch_loss, batch_acc, batch_num = get_metric(model, loss_func, triples)
        total_loss = total_loss + batch_loss.item()
        total_acc, total_num = total_acc + batch_acc, total_num + batch_num
    return total_loss / total_num, total_acc / total_num


def fit(name, max_epoch, embed_mat, path_feats, detail):
    tensors = tensorize(load_feat(path_feats), device)
    bound = int(len(tensors) / 2)
    train_loader, dev_loader = get_loader(tensors[:bound]), get_loader(tensors[bound:])
    embed_mat = torch.Tensor(embed_mat)
    arch = map_item(name, archs)
    model = arch(embed_mat).to(device)
    loss_func = NLLLoss(ignore_index=0, reduction='sum')
    learn_rate, min_rate = 1e-3, 1e-5
    min_dev_loss = float('inf')
    trap_count, max_count = 0, 3
    print('\n{}'.format(model))
    train, epoch = True, 0
    while train and epoch < max_epoch:
        epoch = epoch + 1
        model.train()
        optim = Adam(model.parameters(), lr=learn_rate)
        start = time.time()
        train_loss, train_acc = batch_train(model, loss_func, optim, train_loader, detail)
        delta = time.time() - start
        with torch.no_grad():
            model.eval()
            dev_loss, dev_acc = batch_dev(model, loss_func, dev_loader)
        extra = ''
        if dev_loss < min_dev_loss:
            extra = ', val_loss reduce by {:.3f}'.format(min_dev_loss - dev_loss)
            min_dev_loss = dev_loss
            trap_count = 0
            torch.save(model, map_item(name, paths))
        else:
            trap_count = trap_count + 1
            if trap_count > max_count:
                learn_rate = learn_rate / 10
                if learn_rate < min_rate:
                    extra = ', early stop'
                    train = False
                else:
                    extra = ', learn_rate divide by 10'
                    trap_count = 0
        epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra)


if __name__ == '__main__':
    path_feats = dict()
    path_feats['sent1_train'] = 'feat/sent1_train.pkl'
    path_feats['sent2_train'] = 'feat/sent2_train.pkl'
    path_feats['label_train'] = 'feat/label_train.pkl'
    path_feats['sent1_dev'] = 'feat/sent1_dev.pkl'
    path_feats['sent2_dev'] = 'feat/sent2_dev.pkl'
    path_feats['label_dev'] = 'feat/label_dev.pkl'
    fit('ptr', 50, embed_mat, path_feats, detail)
