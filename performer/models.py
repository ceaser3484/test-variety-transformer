import torch.nn as nn
import torch
import math

class RoPEEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        half_dim = self.d_model // 2
        freq = torch.exp(-torch.arange(0, half_dim, dtype=torch.float32) / half_dim * math.log(10000)).to(x.device)
        pos = torch.arange(seq_len, dtype=torch.float32).to(x.device)
        angles = pos[:, None] * freq[None, :]
        sin, cos = angles.sin(), angles.cos()

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated.reshape(x.size())

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_dim, r_value):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.d_model = n_head * head_dim
        self.r_value = r_value

        self.q_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_weight = nn.Linear(self.d_model, self.d_model, bias=False)

    def generate_orf_matrix(self, r, d):
        G = torch.randn(r, d)
        Q, _ = torch.linalg.qr(G)
        return Q

    def orthogonal_random_features(self, x, omega, b):
        projection = x @ omega.T + b
        return torch.sqrt(torch.tensor(2.0).to(x.device) / len(b)) * torch.cos(projection)

    def forward(self, pre_query, pre_key, pre_value):
        
        L, d = pre_query.size(1), pre_query.size(2)

        query = self.q_weight(pre_query)
        key = self.k_weight(pre_key)
        value = self.v_weight(pre_value)

        omega = self.generate_orf_matrix(self.r_value, self.d_model).to(query.device)
        b = torch.rand((self.r_value,), device=query.device) * 2 * torch.pi
        
        query_prime = self.orthogonal_random_features(query, omega, b)
        key_prime = self.orthogonal_random_features(key, omega, b)
        print(query_prime.size(), key_prime.size())
        exit()

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.normalization = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.linear2 = nn.Linear(d_model * 4, d_model)

    def forward(self, x):
        x = self.normalization(x)
        x = self.linear1(x)
        x = torch.gelu(x)
        x = self.linear2(x)
        return x


class Encoder(nn.Module):
    def __init__(self, n_head, head_dim, r_value):
        super(Encoder, self).__init__()
        self.attention = MultiHeadAttention(n_head, head_dim, r_value)
        self.ffn = FeedForwardNetwork(n_head * head_dim)

    def forward(self, sequence, mask=None):
        attention_output = self.attention(sequence, sequence, sequence)
        ffn_output = self.ffn(attention_output)
        return ffn_output

class Decoder(nn.Module):
    def __init__(self, n_head, head_dim, r_value):
        super(Decoder, self).__init__()

    def forward(self, sequence, encoder_output, mask=None):
        pass



class Performer(nn.Module):
    def __init__(self, hyper_params, vocab_size):
        super(Performer, self).__init__()

        d_model = hyper_params['n_head'] * hyper_params['head_dim']
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = RoPEEmbedding(d_model)
        self.encoder = Encoder(hyper_params['n_head'], hyper_params['head_dim'], hyper_params['r_value'])
        self.decoder = Decoder(hyper_params['n_head'], hyper_params['head_dim'], hyper_params['r_value'])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, sequence):
        pre_sequence = self.embedding(sequence) # (batch, seq_len, d_model)
        sequence = pre_sequence + self.pos_embedding(pre_sequence)
        encoder_output = self.encoder(sequence)
        
        exit()