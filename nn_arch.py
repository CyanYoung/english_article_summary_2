import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Ptr(nn.Module):
    def __init__(self, embed_mat):
        super(Ptr, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.eps = 1e-10
        self.embed = nn.Embedding(vocab_num, embed_len, _weight=embed_mat)
        self.encode = nn.GRU(embed_len, 200, batch_first=True)
        self.decode = nn.GRU(embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.gate = nn.Linear(400, 1)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, vocab_num))

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
        p1 = (1 - g + self.eps) * a
        p2 = (g + self.eps) * F.softmax(self.dl(s2), dim=-1)
        return torch.cat((p2, p1), dim=-1)


class PtrEncode(nn.Module):
    def __init__(self, embed_mat):
        super(PtrEncode, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len)
        self.encode = nn.GRU(embed_len, 200, batch_first=True)

    def forward(self, x):
        x = self.embed(x)
        h1, h1_n = self.encode(x)
        return h1


class PtrDecode(nn.Module):
    def __init__(self, embed_mat):
        super(PtrDecode, self).__init__()
        self.eps = 1e-10
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len)
        self.decode = nn.GRU(embed_len, 200, batch_first=True)
        self.query, self.key, self.val = [nn.Linear(200, 200)] * 3
        self.gate = nn.Linear(400, 1)
        self.dl = nn.Sequential(nn.Dropout(0.2),
                                nn.Linear(400, vocab_num))

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
        s2 = torch.cat((h2, c), dim=-1)
        g = torch.sigmoid(self.gate(s2))
        p1 = (1 - g + self.eps) * a
        p2 = (g + self.eps) * F.softmax(self.dl(s2), dim=-1)
        return torch.cat((p2, p1), dim=-1)


class PtrPlot(nn.Module):
    def __init__(self, embed_mat):
        super(PtrPlot, self).__init__()
        vocab_num, embed_len = embed_mat.size()
        self.embed = nn.Embedding(vocab_num, embed_len)
        self.encode = nn.GRU(embed_len, 200, batch_first=True)
        self.decode = nn.GRU(embed_len, 200, batch_first=True)
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
