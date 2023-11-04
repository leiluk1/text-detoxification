import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
import math


class PositionalEncoding(nn.Module):
    """
    Helper Module that adds positional encoding to the token embedding 
    to introduce a notion of word order.
    """
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 128):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor) -> Tensor:
            """
            Passes the token embeddings through the transformer encoder layers.

            Args:
                token_embedding (Tensor): The input token embedding.

            Returns:
                Tensor: The output tensor after passing through the transformer encoder layers.
            """
            return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    """
    Helper Module to convert tensor of input indices into corresponding tensor 
    of token embeddings.
    """
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
            """
            Convert input tokens into corresponding tensor of token embeddings.

            Args:
                tokens (Tensor): Input tensor of tokens.

            Returns:
                Tensor: Output tensor of token embeddings.
            """
            return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2SeqTransformer(nn.Module):
    """
    A sequence-to-sequence transformer model for text detoxification.
    """
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, vocab_size)
        self.src_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor) -> Tensor:
        """
        Passes the source and target sequences through the transformer and returns the output.

        Args:
            src (Tensor): The source sequence.
            trg (Tensor): The target sequence.
            src_mask (Tensor): The mask for the source sequence.
            tgt_mask (Tensor): The mask for the target sequence.
            src_padding_mask (Tensor): The padding mask for the source sequence.
            tgt_padding_mask (Tensor): The padding mask for the target sequence.
            memory_key_padding_mask (Tensor): The padding mask for the memory.

        Returns:
            Tensor: The output of the transformer.
        """
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor) -> Tensor:
        """
        Passes the source sequence through the transformer encoder and returns the output.

        Args:
            src (Tensor): The source sequence.
            src_mask (Tensor): The mask for the source sequence.

        Returns:
            Tensor: The output of the transformer encoder.
        """
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor) -> Tensor:
        """
        Passes the target sequence through the transformer decoder and returns the output.

        Args:
            tgt (Tensor): The target sequence.
            memory (Tensor): The memory.
            tgt_mask (Tensor): The mask for the target sequence.

        Returns:
            Tensor: The output of the transformer decoder.
        """
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)
