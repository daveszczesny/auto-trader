import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    
    def forward(self, x):
        x = self.embedding(x)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = x.permute(1, 0, 2)
        transformer_out = self.transformer(x, x)
        out = self.fc(transformer_out[:, -1, :])
        return out