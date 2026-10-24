import torch
import torch.nn as nn

class ByteStudentModel(nn.Module):
    def __init__(self, vocab_size=256, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.to_logits = nn.Linear(d_model, vocab_size)

    def forward(self, byte_ids):
        # byte_ids shape: [batch, seq_len]
        b, t = byte_ids.shape
        
        # Causal mask so future bytes cannot be seen during attention
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=byte_ids.device)
        
        x = self.embed(byte_ids)
        x = self.transformer(x, mask=mask, is_causal=True)
        logits = self.to_logits(x)
        
        # Output shape: [batch, seq_len, 256]
        return logits