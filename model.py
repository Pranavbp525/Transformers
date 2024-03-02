import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InputEmbeddings(nn.Module):
    """
    Embeds input tokens using learned embeddings and scales them according to the model size.

    Parameters:
    - d_model (int): The dimensionality of the model's embeddings.
    - vocab_size (int): The size of the vocabulary.

    Methods:
    - forward(x): Embeds input tokens and scales the embeddings.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEmbeddings(nn.Module):
    """
    Applies positional encoding to the input embeddings to inject some information about the relative
    or absolute position of the tokens in the sequence.

    Parameters:
    - d_model (int): The dimensionality of the model's embeddings.
    - seq_len (int): The maximum sequence length that this module will encounter.
    - dropout (float): The dropout rate.

    Methods:
    - forward(x): Applies positional encoding to the input embeddings.
    """

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(self.seq_len, self.d_model)
        position = torch.arange(0, self.seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(1000.0) / self.d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).detach()
        x = self.dropout(x)

        return x


class LayerNormalization(nn.Module):
    """
    Applies Layer Normalization over a mini-batch of inputs.

    Parameters:
    - eps (float): A value added to the denominator for numerical stability.

    Methods:
    - forward(x): Applies layer normalization on the input.
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdims=True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):
    """
    Implements the feedforward block of the transformer architecture, consisting of two linear
    transformations with a ReLU activation in between.

    Parameters:
    - d_model (int): The dimensionality of the input and output.
    - d_ff (int): The dimensionality of the hidden layer.
    - dropout (float): The dropout rate.

    Methods:
    - forward(x): Applies two linear transformations with ReLU activation and dropout.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class MultiHeadAttention(nn.Module):
    """
    Implements the multi-head attention mechanism.

    Parameters:
    - d_model (int): The dimensionality of the input and output.
    - num_heads (int): The number of attention heads.
    - dropout (float): The dropout rate.

    Methods:
    - forward(q, k, v, mask=None): Computes the multi-head attention for query, key, and value tensors.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads

        # Ensure the input and output dimensions are compatible with d_model
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        if dropout is not None:
            scores = dropout(F.softmax(scores, dim=-1))

        return torch.matmul(scores, value)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # Linear projections in batch from d_model => h x d_k
        q = self.W_q(q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(k).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(v).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention on all the projected vectors in batch.
        x = self.attention(q, k, v, self.dropout)

        # "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        return self.W_o(x)


class ResidualConnection(nn.Module):
    """
    Implements a residual connection followed by layer normalization.

    Parameters:
    - size (int): The dimensionality of the input and output.
    - dropout (float): The dropout rate.

    Methods:
    - forward(x, sublayer): Applies the residual connection to any sublayer with the same size.
    """
    def __init__(self, size, dropout: float = 0.1):
        super().__init__()
        self.norm = LayerNormalization(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # Apply normalization, then the sublayer function, and finally dropout
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Represents one block of the encoder structure in the transformer architecture, including multi-head
    attention and feedforward network with residual connections and layer normalization.

    Parameters:
    - d_model (int): The dimensionality of the input and output.
    - num_heads (int): The number of attention heads.
    - d_ff (int): The dimensionality of the hidden layer in the feedforward network.
    - dropout (float): The dropout rate.

    Methods:
    - forward(x, src_mask): Processes one block of the encoder.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_conn_1 = ResidualConnection(d_model, dropout)
        self.residual_conn_2 = ResidualConnection(d_model, dropout)

    def forward(self, x, src_mask):
        x = self.residual_conn_1(x, lambda x: self.self_attention(x, x, x, src_mask))
        x = self.residual_conn_2(x, self.feed_forward)
        return x


class DecoderBlock(nn.Module):
    """
    Represents one block of the decoder structure in the transformer architecture, including two multi-head
    attention layers and one feedforward network with residual connections and layer normalization.

    Parameters:
    - d_model (int): The dimensionality of the input and output.
    - num_heads (int): The number of attention heads.
    - d_ff (int): The dimensionality of the hidden layer in the feedforward network.
    - dropout (float): The dropout rate.

    Methods:
    - forward(x, encoder_output, src_mask, tgt_mask): Processes one block of the decoder.
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.residual_conn_1 = ResidualConnection(d_model, dropout)
        self.residual_conn_2 = ResidualConnection(d_model, dropout)
        self.residual_conn_3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_conn_1(x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.residual_conn_2(x, lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask))
        x = self.residual_conn_3(x, self.feed_forward)
        return x


class Encoder(nn.Module):
    """
    Represents the encoder part of the transformer architecture, consisting of a stack of N encoder blocks.

    Parameters:
    - N (int): The number of encoder blocks.
    - d_model (int): The dimensionality of the input and output.
    - num_heads (int): The number of attention heads.
    - d_ff (int): The dimensionality of the hidden layer in the feedforward network.
    - dropout (float): The dropout rate.

    Methods:
    - forward(x, mask): Processes the input through N encoder blocks.
    """
    def __init__(self, N: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
    Represents the decoder part of the transformer architecture, consisting of a stack of N decoder blocks.

    Parameters:
    - N (int): The number of decoder blocks.
    - d_model (int): The dimensionality of the input and output.
    - num_heads (int): The number of attention heads.
    - d_ff (int): The dimensionality of the hidden layer in the feedforward network.
    - dropout (float): The dropout rate.

    Methods:
    - forward(x, encoder_output, src_mask, tgt_mask): Processes the input through N decoder blocks.
    """
    def __init__(self, N: int, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(N)])
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    """
    Implements the Transformer architecture for sequence to sequence learning.

    Parameters:
    - src_vocab_size (int): Size of the source vocabulary.
    - tgt_vocab_size (int): Size of the target vocabulary.
    - src_seq_len (int): Maximum sequence length for the source.
    - tgt_seq_len (int): Maximum sequence length for the target.
    - N (int): The number of blocks in the encoder and decoder.
    - d_model (int): The dimensionality of the input and output.
    - num_heads (int): The number of attention heads.
    - d_ff (int): The dimensionality of the hidden layer in the feedforward network.
    - dropout (float): The dropout rate.

    Methods:
    - encode(src, src_mask): Encodes the source input.
    - decode(tgt, encoder_output, src_mask, tgt_mask): Decodes the target input.
    - forward(src, tgt, src_mask, tgt_mask): Processes the input through the entire model.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int,
                 N: int = 6, d_model: int = 512, num_heads: int = 8, d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.src_embedding = InputEmbeddings(d_model, src_vocab_size)
        self.tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)
        self.src_pe = PositionalEmbeddings(d_model, src_seq_len, dropout)
        self.tgt_pe = PositionalEmbeddings(d_model, tgt_seq_len, dropout)
        self.encoder = Encoder(N, d_model, num_heads, d_ff, dropout)
        self.decoder = Decoder(N, d_model, num_heads, d_ff, dropout)
        self.linear = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pe(src)
        return self.encoder(src, src_mask)

    def project(self, x):
        x = self.linear(x)
        return torch.log_softmax(x, dim=-1)

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pe(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask):
        src = self.encode(src, src_mask)
        tgt = self.decode(tgt, src, src_mask, tgt_mask)
        x = self.project(tgt)
        return x
