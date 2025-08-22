import torch
import torch.nn as nn
import math

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

    def forward(self, sentence):
        batch_size, seq_length = sentence.size()
        return self.encodding[:seq_length, :]

class LinformerSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, head_dim: int, e_weight, f_weight, dropout):
        super(LinformerSelfAttention, self).__init__()

        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = head_dim

        self.e_weight = e_weight
        self.f_weight = f_weight

        self.query_weight = nn.Linear(self.d_model, self.d_model)
        self.key_weight = nn.Linear(self.d_model, self.d_model)
        self.value_weight = nn.Linear(self.d_model, self.d_model)

        self.out_proj = nn.Linear(self.d_model, self.d_model)

        self.normalization = nn.LayerNorm(self.d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=False):

        origin_sequence = query.clone()
        batch_size, seq_len, _ = query.size()

        e_weight_sliced = self.e_weight[:, :seq_len]
        f_weight_sliced = self.f_weight[:, :seq_len]
        
        k_key = e_weight_sliced @ key
        k_value = f_weight_sliced @ value

        query = self.query_weight(query)
        key = self.key_weight(k_key)
        value = self.value_weight(k_value)

        query = query.view(batch_size, self.n_head, -1, self.head_dim)
        key = key.view(batch_size, self.n_head, -1, self.head_dim)
        value = value.view(batch_size, self.n_head, -1, self.head_dim)

        pre_attention = torch.matmul(query, torch.transpose(key, -1, -2)) / math.sqrt(self.head_dim)
        
        pre_attention = torch.nn.functional.softmax(pre_attention, dim=-1)

        attention = torch.matmul(pre_attention, value)
        attention = attention.view(batch_size, -1, self.d_model)

        attention = self.out_proj(attention)
        attention = self.dropout(attention)
        attention = self.normalization(attention + origin_sequence)

        return attention


class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super(FeedForwardNetwork, self).__init__()
        self.feed_forward_network = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model)
        )
        self.normalization = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        origin_sequence = sequence.clone()
        sequence = self.dropout(self.feed_forward_network(sequence))
        sequence = self.normalization(sequence + origin_sequence)
        return sequence


class Blocks(nn.Module):
    def __init__(self, hyper_parameter):
        super().__init__()

        d_model = hyper_parameter['n_head'] * hyper_parameter['head_dim']

        # self.E_weight = nn.Linear(hyper_parameter['max_len'], hyper_parameter['k'], bias=False)
        # self.F_weight = nn.Linear(hyper_parameter['max_len'], hyper_parameter['k'], bias=False)
        self.E_weight = nn.Parameter(torch.randn(hyper_parameter['k'], hyper_parameter['max_len']))
        self.F_weight = nn.Parameter(torch.randn(hyper_parameter['k'], hyper_parameter['max_len']))

        self.linformer_self_attention = LinformerSelfAttention(d_model, hyper_parameter['n_head'],
                                                               hyper_parameter['head_dim'], self.E_weight, self.F_weight, hyper_parameter['attention_dropout'])

        self.feed_forward_network = FeedForwardNetwork(d_model, hyper_parameter['ffn_dropout'])

    def forward(self, sequence):

        attention = self.linformer_self_attention(sequence, sequence, sequence)
        result = self.feed_forward_network(attention)
        return result


class Linformer(nn.Module):
    def __init__(self, hyper_parameter, num_vocab, device):
        super(Linformer, self).__init__()
        max_len = hyper_parameter['max_len']
        n_stack = hyper_parameter['n_stack']
        d_model = hyper_parameter['n_head'] * hyper_parameter['head_dim']

        self.embedding = nn.Embedding(num_vocab, d_model)

        self.positional_encoding = PositionalEncoding(d_model, max_len, device)

        self.blocks = nn.ModuleList([Blocks(hyper_parameter) for _ in range(n_stack)])

        self.prob_linear = nn.Linear(d_model, num_vocab)
        self.dropout = nn.Dropout(hyper_parameter['embedding_dropout'])

    def forward(self, sequence):
        sequence = self.dropout(self.embedding(sequence) + self.positional_encoding(sequence))

        for block in self.blocks:
            sequence = block(sequence)

        result = self.prob_linear(sequence)
        return result

class LinformerEncoderBlock(nn.Module):
    def __init__(self, hyper_parameter):
        d_model = hyper_parameter['n_head'] * hyper_parameter['head_dim']

        # self.E_weight = nn.Linear(hyper_parameter['max_len'], hyper_parameter['k'], bias=False)
        # self.F_weight = nn.Linear(hyper_parameter['max_len'], hyper_parameter['k'], bias=False)
        self.E_weight = nn.Parameter(torch.randn(hyper_parameter['max_len'], hyper_parameter['k']))
        self.F_weight = nn.Parameter(torch.randn(hyper_parameter['max_len'], hyper_parameter['k']))

        self.linformer_self_attention = LinformerSelfAttention(d_model, hyper_parameter['n_head'],
                                                               hyper_parameter['head_dim'], self.E_weight, self.F_weight, hyper_parameter['attention_dropout'])

        self.feed_forward_network = FeedForwardNetwork(d_model, hyper_parameter['ffn_dropout'])

    def forward(self, sequence):

        attention = self.linformer_self_attention(sequence, sequence, sequence)
        result = self.feed_forward_network(attention)
        return result

class LinforemrEncoder(nn.Module):
    pass

class LinformerDecoder(nn.Module):
    pass

class LinformerGenerator(nn.Module):
    pass