import torch
import torch.nn as nn
import math


# implementation is based on https://arxiv.org/abs/1706.03762
# explainer of article is: Umar Jamil https://www.youtube.com/watch?v=ISNdQcPhsts


class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        :param d_model: dimension of embedding
        :param vocab_size: size of vocabulary
        """
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # so the idea of Embedding in this case is to map position of word and embedding of this position
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        """
        :param x: position of word
        :return: vector representation of word x in vocabulary
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout: float):
        """
        :param d_model: dimension of embedding
        :param seq_len: maximum length of sequence
        :param dropout: param used to less overfit
        """

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Calculation of positional embedding from article

        # create a matrix of shpae (seq_len, d_model)
        pe = torch.zero_(seq_len, d_model)
        # create a vector of shape (seq_len) before unsqueeze
        position = torch.arange(0, seq_len - 1, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # take a log for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # this construction is used when you don't want your param be learned but saved in model file
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)

        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10 ** -6):
        """

        :param eps: Epsilon used to avoid division by zero
        """
        super().__init__()
        self.eps = eps
        # nn.Parameter makes variable learnable
        self.alpha = nn.Parameter(torch.ones(1))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(1))  # Added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # dim = -1 means last dimension
        std = x.std(dim=-1, keepdim=True)

        # That is the implementation of LayerNorm
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        """

        :param d_model: dimension of embedding
        :param d_ff: dimension of inner layer
        :param dropout: param used to less overfit
        """
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # W1 and B1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)  # W2 and B2

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_1(
            self.dropout(
                torch.relu(
                    self.linear_1(x)
                )
            )
        )


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        """
        :param d_model: dimension of embedding
        :param h: number of heads
        :param dropout: param used to less overfit
        """
        super().__init__()
        self.d_model = d_model
        self.h = h
        # d_model should be divisible by h, because if it's not, we won't be able to equally separate embedding onto
        # these amount of heads
        assert d_model % h == 0, 'd_model is not divisible by h'
        self.d_k = d_model // h
        # Query, Key, Value weights
        self.w_q = nn.Linear(in_features=d_model, out_features=d_model)  # Wq
        self.w_k = nn.Linear(in_features=d_model, out_features=d_model)  # Wk
        self.w_v = nn.Linear(in_features=d_model, out_features=d_model)  # Wv

        # Wo is the output matrix of shape (h * d_v, d_model), but d_v is the same as d_k
        # so its shape is (d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            attention_scores.masked_fill(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)  # (Batch, h, seq_len, seq_len)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # NOTE about mask: if we don't want some words to interact with others
        # we can mask them, thereby make them invisible

        query = self.w_q(q)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        key = self.w_k(k)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        value = self.w_v(v)  # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)

        # (batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Compute number of heads
        x, self.attention_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout)

        # Concatenate them

        # (Batch, h, seq_len, d_k) --> (Batch, seq_len, h, d_k) --> (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)

