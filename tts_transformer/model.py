import math
import torch
import torch.nn as nn

################################################################
# PositionalEncoding
################################################################
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x shape: [seq_len, batch, d_model]
        """
        seq_len = x.size(0)
        if seq_len > self.max_len:
            device = x.device
            # Tạo positional encoding mới nếu seq_len > max_len
            pe = torch.zeros(seq_len, self.d_model, device=device)
            position = torch.arange(0, seq_len, dtype=torch.float32, device=device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=device).float()
                                 * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(1)
            x = x + pe
        else:
            x = x + self.pe[:seq_len]
        return self.dropout(x)

################################################################
# TransformerEncoder
################################################################
class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerEncoder, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        return self.transformer_encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)

################################################################
# TransformerDecoder
################################################################
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super(TransformerDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        return self.transformer_decoder(
            tgt, memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask
        )

################################################################
# TransformerTTS
################################################################
class TransformerTTS(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=256, nhead=4, num_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        """
        input_dim: kích thước vocab (số ký tự)
        output_dim: số băng tần mel
        """
        super(TransformerTTS, self).__init__()
        self.d_model = d_model

        self.text_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        self.mel_embed = nn.Linear(output_dim, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len=5000, dropout=dropout)

        self.encoder = TransformerEncoder(d_model, nhead, num_layers, dim_feedforward, dropout)
        self.decoder = TransformerDecoder(d_model, nhead, num_layers, dim_feedforward, dropout)

        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None):
        """
        src shape: [batch, text_len]
        tgt shape: [batch, n_mels, mel_len]
        """
        # Embed text
        src_embedded = self.text_embedding(src) * math.sqrt(self.d_model)
        src_embedded = src_embedded.permute(1, 0, 2)  # [text_len, batch, d_model]
        src_embedded = self.pos_encoder(src_embedded)

        # Embed mel
        tgt_embedded = self.mel_embed(tgt.permute(0, 2, 1)) * math.sqrt(self.d_model)
        tgt_embedded = tgt_embedded.permute(1, 0, 2)  # [mel_len, batch, d_model]
        tgt_embedded = self.pos_decoder(tgt_embedded)

        # Encode
        memory = self.encoder(src_embedded, src_mask, src_key_padding_mask)
        # Decode
        output = self.decoder(
            tgt_embedded, memory,
            tgt_mask, memory_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask
        )

        # Projection => [mel_len, batch, n_mels] => permute => [batch, n_mels, mel_len]
        output = self.fc_out(output)
        output = output.permute(1, 2, 0)

        return output