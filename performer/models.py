import torch.nn as nn
import torch
from math import log

class RoPEEmbedding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        half_dim = self.d_model // 2
        freq = torch.exp(-torch.arange(0, half_dim, dtype=torch.float32) / half_dim * log(10000)).to(x.device)
        pos = torch.arange(seq_len, dtype=torch.float32).to(x.device)
        angles = pos[:, None] * freq[None, :]
        sin, cos = angles.sin(), angles.cos()

        x1, x2 = x[..., ::2], x[..., 1::2]
        x_rotated = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rotated.reshape(x.size())

class MultiHeadAttention(nn.Module):
    def __init__(self, n_head, head_dim, r_value, dropout, causal_mask=False):
        super().__init__()
        self.n_head = n_head
        self.head_dim = head_dim
        self.d_model = n_head * head_dim
        self.r_value = r_value
        self.causal_mask = causal_mask

        self.q_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_weight = nn.Linear(self.d_model, self.d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(self.d_model)

    def generate_orf_matrix(self, r, d):
        G = torch.randn(r, d)
        Q, _ = torch.linalg.qr(G)
        return Q

    def orthogonal_random_features(self, x, omega, b):
        projection = x @ omega.T + b
        return torch.sqrt(torch.tensor(2.0).to(x.device) / len(b)) * torch.cos(projection)

    def forward(self, pre_query, pre_key, pre_value):
        
        batch_size, L, d_model = pre_query.size()

        query = self.q_weight(pre_query).view(batch_size, L, self.n_head, self.head_dim)
        key = self.k_weight(pre_key).view(batch_size, L, self.n_head, self.head_dim)
        value = self.v_weight(pre_value).view(batch_size, L, self.n_head, self.head_dim)

        # print(f"""query: {query.size()}
        # key: {key.size()}
        # value: {value.size()}""")

        omega = self.generate_orf_matrix(self.r_value, self.head_dim).to(pre_query.device)
        # print("omega size:", omega.size())

        b = torch.rand(self.r_value, device=pre_query.device) * 2 * torch.pi
        # print("b size:", b.size())

        query_prime = self.orthogonal_random_features(query, omega, b)
        key_prime = self.orthogonal_random_features(key, omega, b)

        # print("query_prime size: ", query_prime.size())
        # print("key_prime size: ", key_prime.size())
        
        if self.causal_mask:
            # print("decoder attention")

            key_prefix = torch.cumsum(key_prime, dim=1)
            # print("key_prefix size: ", key_prefix.size())

            kv = key_prime.unsqueeze(-1) * value.unsqueeze(-2)
            # print("kv size: ", kv.size())

            kv_prefix = torch.cumsum(kv, dim=1)
            # print("kv_prefix size: ", kv_prefix.size())

            numerator = (query_prime.unsqueeze(-1) * kv_prefix).sum(dim=-2)
            # print("numerator size: ", numerator.size())

            denominator = (query_prime * key_prefix).sum(dim=-1, keepdim=True)
            denominator = torch.clamp(denominator, min=1e-6)  # 작은 값으로 클램프하여 안정성 확보
            # print("denominator size: ", denominator.size())

        else:
            # print("\nencoder attention")
            kv = torch.matmul(key_prime.permute(0,2,3,1), value.transpose(1,2))
            # print("kv size: ", kv.size())

            numerator = torch.matmul(query_prime.transpose(1,2), kv)
            # print("numerator size: ", numerator.size())

            k_prime_sum = key_prime.sum(dim=1)
            # print("k_prime_sum size: ", k_prime_sum.size())
            
            # print("key_prime_sum add dim", k_prime_sum.unsqueeze(-1).size())
            denominator = torch.matmul(query_prime.transpose(1,2), k_prime_sum.unsqueeze(-1))
            denominator = torch.clamp(denominator, min=1e-6)  # 작은 값으로 클램프하여 안정성 확보
            # print("denominator size: ", denominator.size())

        out = numerator / (denominator + 1e-6)
        out = out.view(batch_size, L, d_model)
        # print("out size: ", out.size())
        pre_result = self.out_weight(out)
        result = self.norm(pre_result + pre_query)
        # print("result size: ", result.size())
        return result

class FeedForwardNetwork(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.normalization = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_model * 4)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(d_model * 4, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence):
        x = self.linear1(sequence)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.normalization(x + sequence)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, n_head, head_dim, r_value, att_dropout, ffn_dropout):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(n_head, head_dim, r_value, att_dropout, causal_mask=False)
        self.ffn = FeedForwardNetwork(n_head * head_dim, ffn_dropout)

    def forward(self, sequence):
        attention_output = torch.utils.checkpoint.checkpoint(self.attention, sequence, sequence, sequence)
        ffn_output = torch.utils.checkpoint.checkpoint(self.ffn, attention_output)
        return ffn_output

class DecoderBlock(nn.Module):
    def __init__(self, n_head, head_dim, r_value, att_dropout, ffn_dropout):
        super(DecoderBlock, self).__init__()
        self.attention_1 = MultiHeadAttention(n_head, head_dim, r_value, att_dropout, causal_mask=True)
        self.attention_2 = MultiHeadAttention(n_head, head_dim, r_value, att_dropout, causal_mask=False)
        self.fnn = FeedForwardNetwork(n_head * head_dim, ffn_dropout)

    def forward(self, sequence, encoder_output):
        attention_output_1 = torch.utils.checkpoint.checkpoint(self.attention_1, sequence, sequence, sequence)
        attention_output_2 = torch.utils.checkpoint.checkpoint(self.attention_2, attention_output_1, encoder_output, encoder_output)
        ffn_output = torch.utils.checkpoint.checkpoint(self.fnn, attention_output_2)
        return ffn_output

class Encoder(nn.Module):
    def __init__(self, n_head, head_dim, r_value, n_stack, att_dropout, ffn_dropout):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList([EncoderBlock(n_head, head_dim, r_value, att_dropout, ffn_dropout) for _ in range(n_stack)])
    
    def forward(self, sequence):
        for block in self.blocks:
            sequence = block(sequence)
        return sequence

class Decoder(nn.Module):
    def __init__(self, n_head, head_dim, r_value, n_stack, att_dropout, ffn_dropout):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList([DecoderBlock(n_head, head_dim, r_value, att_dropout, ffn_dropout) for _ in range(n_stack)])
    
    def forward(self, sequence, encoder_output):
        for block in self.blocks:
            sequence = block(sequence, encoder_output)
        return sequence

class Performer(nn.Module):
    def __init__(self, hyper_params, vocab_size):
        super(Performer, self).__init__()

        d_model = hyper_params['n_head'] * hyper_params['head_dim']
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = RoPEEmbedding(d_model)
        self.dropout = nn.Dropout(hyper_params['embedding_dropout'])
        self.encoder = Encoder(hyper_params['n_head'], hyper_params['head_dim'], hyper_params['r_value'], hyper_params['n_stack'],
                               hyper_params['attention_dropout'], hyper_params['ffn_dropout'])
        self.decoder = Decoder(hyper_params['n_head'], hyper_params['head_dim'], hyper_params['r_value'], hyper_params['n_stack'],
                               hyper_params['attention_dropout'], hyper_params['ffn_dropout'])
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, sequence):
        pre_sequence = self.embedding(sequence) # (batch, seq_len, d_model)
        sequence = pre_sequence + self.pos_embedding(pre_sequence)
        sequence = self.dropout(sequence)
        encoder_output = self.encoder(sequence)
        decoder_output = self.decoder(sequence, encoder_output)
        output = self.output_layer(decoder_output)
        return output