# transformer.py

import time
import torch
import torch.nn as nn
import numpy as np
import random
import math
from torch import optim
import matplotlib.pyplot as plt
from typing import List
from utils import *

class LetterCountingExample(object):
    def __init__(self, input: str, output: np.array, vocab_index: Indexer):
        self.input = input
        self.input_indexed = np.array([vocab_index.index_of(ci) for ci in input])
        self.input_tensor = torch.LongTensor(self.input_indexed)
        self.output = output
        self.output_tensor = torch.LongTensor(self.output)

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

class Transformer(nn.Module):
    def __init__(self, vocab_size, num_positions, d_model, d_internal, num_classes, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, num_positions)
        self.layers = nn.ModuleList([TransformerLayer(d_model, d_internal) for _ in range(num_layers)])
        self.output_layer = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, indices, mask=None):
        x = self.embedding(indices)
        x = self.pos_encoding(x)
        attention_maps = []
        for layer in self.layers:
            x, attn_map = layer(x, mask)
            attention_maps.append(attn_map)
        logits = self.output_layer(self.dropout(x))
        return logits, attention_maps

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_internal):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_internal),
            nn.ReLU(),
            nn.Linear(d_internal, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        attn_output, attn_weights = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x, attn_weights

def train_classifier(args, train, dev):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Transformer(
        vocab_size=27,
        num_positions=20,
        d_model=64,
        d_internal=128,
        num_classes=3,
        num_layers=2
    ).to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fcn = nn.CrossEntropyLoss()
    num_epochs = 10

    for t in range(num_epochs):
        loss_this_epoch = 0.0
        random.seed(t)
        ex_idxs = list(range(len(train)))
        random.shuffle(ex_idxs)

        for ex_idx in ex_idxs:
            model.zero_grad()
            ex = train[ex_idx]
            logits, _ = model(ex.input_tensor.unsqueeze(0).to(device))
            
            loss = loss_fcn(logits.squeeze(0), ex.output_tensor.to(device))
            loss.backward()
            optimizer.step()
            loss_this_epoch += loss.item()

        print(f"Epoch {t+1}, Loss: {loss_this_epoch/len(ex_idxs):.4f}")

    return model

####################################
# DO NOT MODIFY IN YOUR SUBMISSION #
####################################

def decode(model: Transformer, dev_examples: List[LetterCountingExample], do_print=False, do_plot_attn=False):
    num_correct = 0
    num_total = 0
    if len(dev_examples) > 100:
        print("Decoding on a large number of examples (%i); not printing or plotting" % len(dev_examples))
        do_print = False
        do_plot_attn = False
    for i in range(0, len(dev_examples)):
        ex = dev_examples[i]
        (logits, attn_maps) = model.forward(ex.input_tensor.unsqueeze(0))
        probs = torch.softmax(logits, dim=-1)
        predictions = np.argmax(probs.squeeze(0).detach().numpy(), axis=1)

        if do_print:
            print("INPUT %i: %s" % (i, ex.input))
            print("GOLD %i: %s" % (i, repr(ex.output.astype(dtype=int))))
            print("PRED %i: %s" % (i, repr(predictions)))

        if do_plot_attn:
            for j in range(0, len(attn_maps)):
                attn_map = attn_maps[j].squeeze(0).detach().numpy()
                fig, ax = plt.subplots()
                im = ax.imshow(attn_map, cmap='hot', interpolation='nearest')
                ax.set_xticks(np.arange(len(ex.input)), labels=ex.input)
                ax.set_yticks(np.arange(len(ex.input)), labels=ex.input)
                ax.xaxis.tick_top()
                plt.savefig("plots/%i_attns%i.png" % (i, j))
        acc = np.sum(predictions == ex.output)  # Correct predictions are counted
        num_correct += int(acc)
        num_total += len(predictions)
    print("Accuracy: %i / %i = %f" % (num_correct, num_total, float(num_correct) / num_total))
