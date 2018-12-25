import math

import torch
import torch.nn as nn

import torch.nn.functional as F


class Ptr(nn.Module):
    def __init__(self, embed_mat):
        super(Ptr, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len, _weight=embed_mat)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.gate = nn.Linear(400, 1)
        self.dla = nn.Sequential(nn.Dropout(0.2),
                                 nn.Linear(400, self.vocab_num),
                                 nn.Softmax(dim=-1))

    def forward(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v)
        s2 = torch.cat((h2, c), dim=-1)
        g = torch.sigmoid(self.gate(s2))
        p2 = self.dla(s2)
        p = torch.cat((g * p2, (1 - g) * a), dim=-1)
        return torch.log(p)


class PtrEncode(nn.Module):
    def __init__(self, embed_mat):
        super(PtrEncode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        return h1


class PtrDecode(nn.Module):
    def __init__(self, embed_mat):
        super(PtrDecode, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.gate = nn.Linear(400, 1)
        self.dla = nn.Sequential(nn.Dropout(0.2),
                                 nn.Linear(400, self.vocab_num),
                                 nn.LogSoftmax(dim=-1))

    def forward(self, y, h1):
        y = self.embed(y)
        h1_n = torch.unsqueeze(h1[:, -1, :], dim=0)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        a = F.softmax(d, dim=-1)
        c = torch.matmul(a, v)
        p1 = torch.log(a)
        s2 = torch.cat((h2, c), dim=-1)
        g = self.gate(s2)
        p2 = self.dla(s2)
        return torch.cat((g * p2, (1 - g) * p1), dim=-1)


class PtrPlot(nn.Module):
    def __init__(self, embed_mat):
        super(PtrPlot, self).__init__()
        self.vocab_num, self.embed_len = embed_mat.size()
        self.embed = nn.Embedding(self.vocab_num, self.embed_len)
        self.encode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.decode = nn.GRU(self.embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3

    def forward(self, x, y):
        x = self.embed(x)
        y = self.embed(y)
        h1, h1_n = self.encode(x)
        h1 = h1[:, :-1, :]
        h2, h2_n = self.decode(y, h1_n)
        q, k, v = self.query(h2), self.key(h1), self.val(h1)
        scale = math.sqrt(k.size(-1))
        d = torch.matmul(q, k.permute(0, 2, 1)) / scale
        return F.softmax(d, dim=-1)
