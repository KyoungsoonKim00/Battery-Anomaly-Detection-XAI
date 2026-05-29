import torch
import torch.nn as nn
from transformers import DistilBertConfig, DistilBertModel
import config

bert_config = DistilBertConfig(
    vocab_size=1, max_position_embeddings=208,
    n_layers=4, n_heads=4, dim=128, hidden_dim=256
)

class SensorTransformerAutoencoder(nn.Module):
    def __init__(self, num_sensors, d_model=128):
        super().__init__()
        self.num_sensors = num_sensors
        self.input_embedding = nn.Linear(1, d_model)
        self.positional_embedding = nn.Embedding(self.num_sensors, d_model)
        self.transformer_encoder = DistilBertModel(bert_config)
        self.decoder_head = nn.Linear(d_model, 1)

    def forward(self, x):
        x_unsqueezed = x.unsqueeze(-1)
        embedded_x = self.input_embedding(x_unsqueezed)
        positions = torch.arange(0, self.num_sensors).expand(x.size(0), -1).to(config.DEVICE)
        pos_emb = self.positional_embedding(positions)
        final_embedding = embedded_x + pos_emb

        padding_size = 208 - self.num_sensors
        attention_mask = torch.ones(x.size(0), self.num_sensors).to(config.DEVICE)

        if padding_size > 0:
            pad = torch.zeros(final_embedding.size(0), padding_size, final_embedding.size(2)).to(config.DEVICE)
            final_embedding = torch.cat([final_embedding, pad], dim=1)
            pad_mask = torch.zeros(x.size(0), padding_size).to(config.DEVICE)
            attention_mask = torch.cat([attention_mask, pad_mask], dim=1)

        encoder_output = self.transformer_encoder(inputs_embeds=final_embedding, attention_mask=attention_mask)
        hidden_state = encoder_output.last_hidden_state
        hidden_state = hidden_state[:, :self.num_sensors, :]
        reconstructed_x = self.decoder_head(hidden_state)
        return reconstructed_x.squeeze(-1)