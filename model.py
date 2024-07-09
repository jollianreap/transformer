import torch
import torch.nn as nn
import math


# implementation is based on https://arxiv.org/abs/1706.03762
# explainer of article is: Umar Jamil https://www.youtube.com/watch?v=ISNdQcPhsts
# dataset:

# TODO
# Add logger in I/O operations
# Write an API
# Create some shell for this project

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
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Calculation of positional embedding from article

        # create a matrix of shpae (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len) before unsqueeze
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  # (seq_len, 1)
        # take a log for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # apply sin and cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        # this construction is used when you don't want your param be learned but saved in model file
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)

        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 10 ** -6):
        """

        :param eps: Epsilon used to avoid division by zero
        """
        super().__init__()
        self.eps = eps
        # nn.Parameter makes variable learnable
        self.alpha = nn.Parameter(torch.ones(features))  # Multiplied
        self.bias = nn.Parameter(torch.zeros(features))  # Added

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
        return self.linear_2(
            self.dropout(
                torch.relu(
                    self.linear_1(x) # ()
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
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)


class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        """
        :param dropout: param used to less overfit
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        return x + self.dropout(
            sublayer(
                self.norm(x)
            )
        )

"""
I faced to problem with comprehension of what decoder and encoder really do
I found this excellent answer:
https://ai.stackexchange.com/questions/41505/which-situation-will-helpful-using-encoder-or-decoder-or-both-in-transformer-mod
"""


class EncoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, feed_forward_block: FeedForwardBlock,
                 dropout: float):
        """
        Encoder block takes as input raw data and process it to create appropriate representations of all the data.
        Further these representations are passed through decoder block to get some answer with length differs from input
        for example machine translation, summarization


        :param self_attention_block: Block used to calculate attention scores
        :param feed_forward_block: Block used as activation block
        :param dropout: param used to less overfit
        """

        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = nn.Dropout(dropout)

        # Module list is used to stack several modules
        self.resudial_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # src mask is the source mask which we want to apply to the input of the encoder
        # for example we don't want padding words to interact with other words

        # MultiheadAttention and First ResudialConnection blocks
        x = self.resudial_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask)) # here we use lambda to put 3 x to MultiHeadAttention block

        x = self.resudial_connections[1](x, self.feed_forward_block)

        return x


class Encoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention,
                 feed_forward_block: FeedForwardBlock, dropout: float):
        """
        Decoder is a block gets as input output of encoder. It changes our data somehow and returns new representations

        :param self_attention_block: Block used to calculate attention scores
        :param cross_attention_block: Block used to calculate attention scores of encoder part and decoder
        :param feed_forward_block: Block used as activation block
        :param dropout: param to less overfit
        """

        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        """
        :param x: decoder input
        :param encoder_output:  encoder output
        :param src_mask: source mask (comes from encoder)
        :param tgt_mask: target mask (comes from decoder)

        """

        # SelfAttention layer
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        # CrossAttention layer (query and values come from encoder)
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        # FeedForward layer
        x = self.residual_connections[2](x, self.feed_forward_block)

        return x


class Decoder(nn.Module):
    def __init__(self, features: int, layers: nn.ModuleList):
        """
        :param layers: list of layers
        """

        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            # Here layer is the DecoderBlock
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return self.norm(x)


class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        """
        ProjectionLayer converts embeddings to positions in vocabulary.
        In article, that's Linear layer

        :param d_model: dimension of embedding
        :param vocab_size: vocabulary size
        """

        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return self.proj(x)


class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: InputEmbedding, tgt_embed: InputEmbedding,
                 src_pos: PositionalEncoding, tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        """

        :param encoder: Encoder block
        :param decoder: Decoder block
        :param src_embed: Embedding on input language
        :param tgt_embed: Embedding on output language
        :param src_pos: Initial positions
        :param tgt_pos: Output positions
        :param projection_layer: Linear layer
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj = projection_layer

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)

        return self.encoder(src, src_mask)

    def project(self, x):
        return self.proj(x)

    def forward(self, x):
        pass


def build_transformer(src_vocab_size: int, tgt_vocab_size: int,
                      src_seq_len: int, tgt_seq_len: int, d_model: int = 512, N: int = 6,
                      h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
    # Create embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    tgt_embed = InputEmbedding(d_model, tgt_vocab_size)

    # Create Positional Encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    # Create encoder and decoder blocks
    encoder_blocks = []
    decoder_blocks = []
    for _ in range(N):
        # Encoder
        encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        # Decoder
        decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block,
                                     feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    # Create encoder and decoder (again via ModuleList)
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # Create the projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    # Create transformer
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed,
        src_pos, tgt_pos, projection_layer
    )

    # Initialize the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)

    return transformer

