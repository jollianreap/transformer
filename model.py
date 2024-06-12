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
        position = torch.arange(0, seq_len-1, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # take a log for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # this construction is used when you don't want your param be learned but saved in model file
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False)

        return self.dropout(x)




