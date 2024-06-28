import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super(PositionalEncoding, self).__init__()

        self.encodding = torch.zeros(max_len, d_model, device=device)
        self.encodding.requires_grad = False

        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, d_model, step=2, device=device).float()

        self.encodding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encodding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # print(self.encodding)

    def forward(self, sentence):
        batch_size, seq_length = sentence.size()
        return self.encodding[:seq_length, :]


class MultiHeaderAttention(nn.Module):
    def __init__(self, d_model, hidden_dim, n_heads, dropout):
        super(MultiHeaderAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        self.query_linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.key_linear = nn.Linear(self.d_model, self.d_model,bias=False)
        self.value_linear = nn.Linear(self.d_model, self.d_model, bias=False)
        self.attention_linear = nn.Linear(self.d_model, self.d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.normalization = nn.LayerNorm(d_model)

    def forward(self, pre_query, pre_key, pre_value, mask=None):

        batch_size, seq_length = pre_query.size(0), pre_query.size(1)
        query = self.query_linear(pre_query)
        key = self.key_linear(pre_key)
        value = self.value_linear(pre_value)

        query = query.view(batch_size, self.n_heads, -1 ,self.hidden_dim)
        key = key.view(batch_size, self.n_heads, -1 ,self.hidden_dim)
        value = value.view(batch_size, self.n_heads, -1 ,self.hidden_dim)

        # query, key, value = self.split(query), self.split(key), self.split(value)

        energy = torch.matmul(query, torch.transpose(key, -1, -2)) / sqrt(self.hidden_dim)

        if mask is not None:
            energy = energy.masked_fill(mask == 1, float('-1e20'))

        energy = F.softmax(energy, dim=-1)

        attention = torch.matmul(energy, value)
        attention = attention.view(batch_size, seq_length, self.d_model)
        attention = self.dropout(self.attention_linear(attention))
        norm_attention = self.normalization(attention + pre_value)

        return norm_attention


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super(FeedForwardNetwork, self).__init__()

        self.feed_forward_network = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

        self.normalization = nn.LayerNorm(d_model)

    def forward(self, sequence):
        ffn_sequence =  self.feed_forward_network(sequence)
        return self.normalization(sequence + ffn_sequence)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, n_heads, dropout):
        super(EncoderBlock, self).__init__()

        self.multi_header_attention = MultiHeaderAttention(d_model, hidden_dim, n_heads, dropout)
        self.feed_forward_network = FeedForwardNetwork(d_model, dropout)

    def forward(self, source):
        attention = self.multi_header_attention(source, source, source)
        ffn_out = self.feed_forward_network(attention)
        return ffn_out


class Encoder(nn.Module):
    def __init__(self, d_model, hidden_dim, n_heads, n_stack, dropout):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([EncoderBlock(d_model, hidden_dim, n_heads, dropout) for _ in range(n_stack)])
        # self.encoder_layers = EncoderBlock(d_model, hidden_dim, n_heads)
    def forward(self, source):
        for encoder in self.encoder_layers:
            source = encoder(source)

        return source


class DecoderBlock(nn.Module):
    def __init__(self, d_model, hidden_dim, n_heads, dropout):
        super(DecoderBlock, self).__init__()
        self.multi_header_attention_1 = MultiHeaderAttention(d_model, hidden_dim, n_heads, dropout)
        self.multi_header_attention_2 = MultiHeaderAttention(d_model, hidden_dim, n_heads, dropout)
        self.feed_forward_network = FeedForwardNetwork(d_model, dropout)

    def forward(self, target, enc_out, target_mask):
        pre_attention = self.multi_header_attention_1(target, target, target, target_mask)
        attention = self.multi_header_attention_2(enc_out, enc_out, pre_attention)
        ffn_out = self.feed_forward_network(attention)

        return ffn_out

class Decoder(nn.Module):
    def __init__(self, d_model, hidden_dim, n_heads, n_stack, trg_vocab_size, dropout, device):
        super(Decoder, self).__init__()
        self.decoder_layers = nn.ModuleList([DecoderBlock(d_model, hidden_dim, n_heads, dropout) for _ in range(n_stack)])
        # self.decoder_layers = DecoderBlock(d_model, hidden_dim, n_heads)
        self.vocab_prob_linear = nn.Linear(d_model, trg_vocab_size)

        self.device = device


    def forward(self, target, enc_out):
        mask = self.make_mask(target)

        for decoder in self.decoder_layers:
            target = decoder(target, enc_out, mask)

        result = self.vocab_prob_linear(target)
        return result

    def make_mask(self, target):
        target_length = target.size(1)
        pre_mask = torch.ones(target_length, target_length)
        mask = torch.triu(pre_mask, diagonal=1)
        return mask.to(self.device)


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, max_len,  hidden_dim, n_heads, n_stack, src_pad_idx, trg_pad_idx, dropout, device='cpu'):
        super(Transformer, self).__init__()
        self.d_model = hidden_dim * n_heads
        self.src_embedding = nn.Embedding(src_vocab_size, self.d_model)
        self.target_embedding = nn.Embedding(trg_vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, max_len, device)
        self.encoder = Encoder(self.d_model, hidden_dim, n_heads, n_stack, dropout)
        self.decoder = Decoder(self.d_model, hidden_dim, n_heads, n_stack, trg_vocab_size, dropout, device)

        self.dropout = nn.Dropout(dropout)

    def forward(self, source, target):
        source = self.dropout(self.src_embedding(source) + self.positional_encoding(source))
        target = self.dropout(self.target_embedding(target) + self.positional_encoding(target))
        enc_out = self.encoder(source)
        dec_out = self.decoder(target, enc_out)
        return dec_out


