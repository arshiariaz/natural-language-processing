# transformer_lm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class LanguageModel(object):
    def get_next_char_log_probs(self, context) -> np.ndarray:
        raise NotImplementedError("Only implemented in subclasses")

    def get_log_prob_sequence(self, next_chars, context) -> float:
        raise NotImplementedError("Only implemented in subclasses")

class UniformLanguageModel(LanguageModel):
    def __init__(self, voc_size):
        self.voc_size = voc_size

    def get_next_char_log_probs(self, context):
        return np.ones([self.voc_size]) * np.log(1.0 / self.voc_size)

    def get_log_prob_sequence(self, next_chars, context):
        return np.log(1.0 / self.voc_size) * len(next_chars)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class NeuralLanguageModel(nn.Module, LanguageModel):
    def __init__(self, vocab_size, num_positions, d_model=256, d_internal=512, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, num_positions)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8, dim_feedforward=d_internal, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size
        self.num_positions = num_positions
        self.vocab_index = None

    def forward(self, input_batch, mask):
        x = self.embedding(input_batch)
        x = self.pos_encoding(x)
        x = self.transformer(x, mask)
        logits = self.output_layer(x)
        return logits

    def get_next_char_log_probs(self, context):
        self.eval()
        context = context[-self.num_positions:].rjust(self.num_positions, ' ')
        context_tensor = torch.tensor(
            [[self.vocab_index.index_of(c) for c in context]],
            dtype=torch.long
        ).to(next(self.parameters()).device)
        
        mask = self.generate_square_subsequent_mask(len(context_tensor[0])).to(context_tensor.device)
        
        with torch.no_grad():
            logits = self(context_tensor, mask)
        
        log_probs = F.log_softmax(logits[0, -1], dim=-1)
        return log_probs.cpu().numpy()

    def get_log_prob_sequence(self, next_chars, context):
        total_log_prob = 0.0
        current_context = context

        for char in next_chars:
            next_char_log_probs = self.get_next_char_log_probs(current_context)
            char_idx = self.vocab_index.index_of(char)
            total_log_prob += next_char_log_probs[char_idx]
            current_context += char
            if len(current_context) > self.num_positions:
                current_context = current_context[-self.num_positions:]
        
        return total_log_prob

    @staticmethod
    def generate_square_subsequent_mask(sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def train_lm(args, train_text, dev_text, vocab_index):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab_size = len(vocab_index)
    seq_length = 20
    model = NeuralLanguageModel(
        vocab_size,
        num_positions=seq_length,
        d_model=256,
        d_internal=512,
        num_layers=4
    )
    model.to(device)
    model.vocab_index = vocab_index
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    batch_size = 64

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0

        for i in range(0, len(train_text) - seq_length, batch_size * seq_length):
            batch_end = min(i + batch_size * seq_length, len(train_text) - seq_length)
            batch_text = train_text[i:batch_end]

            input_batch = []
            target_batch = []

            for j in range(0, len(batch_text) - seq_length, seq_length):
                input_seq = ' ' + batch_text[j:j + seq_length - 1]
                target_seq = batch_text[j:j + seq_length]

                input_batch.append([vocab_index.index_of(c) for c in input_seq])
                target_batch.append([vocab_index.index_of(c) for c in target_seq])

            if not input_batch:
                continue

            input_batch = torch.tensor(input_batch, dtype=torch.long).to(device)
            target_batch = torch.tensor(target_batch, dtype=torch.long).to(device)

            mask = model.generate_square_subsequent_mask(seq_length).to(device)

            optimizer.zero_grad()
            logits = model(input_batch, mask)
            loss = criterion(logits.view(-1, vocab_size), target_batch.view(-1))

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        scheduler.step(avg_loss)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    return model
