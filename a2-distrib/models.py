# models.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sentiment_data import *
from typing import List

class SentimentClassifier(object):
    """
    Sentiment classifier base type
    """

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        Makes a prediction on the given sentence
        :param ex_words: words to predict on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return: 0 or 1 with the label
        """
        raise Exception("Don't call me, call my subclasses")

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        """
        You can leave this method with its default implementation, or you can override it to a batched version of
        prediction if you'd like. Since testing only happens once, this is less critical to optimize than training
        for the purposes of this assignment.
        :param all_ex_words: A list of all exs to do prediction on
        :param has_typos: True if we are evaluating on data that potentially has typos, False otherwise.
        :return:
        """
        return [self.predict(ex_words, has_typos) for ex_words in all_ex_words]

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        """
        :param ex:
        :return: 1, always predicts positive class
        """
        return 1

class DANSentimentClassifier(nn.Module):
    def __init__(self, word_embeddings: WordEmbeddings, hidden_size: int, dropout_rate: float):
        super(DANSentimentClassifier, self).__init__()
        
        self.embed = word_embeddings.get_initialized_embedding_layer(frozen=False)
        embedding_dim = word_embeddings.get_embedding_length()
        
        self.ffnn = nn.Sequential(
            nn.Linear(embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 2)
        )
        
        self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, word_indices: torch.Tensor):
        embedded = self.embed(word_indices)
        mask = (word_indices != 0).float().unsqueeze(-1)
        masked_embed = embedded * mask
        averaged = masked_embed.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        logits = self.ffnn(averaged)
        log_probs = self.log_softmax(logits)
        return log_probs

class PrefixEmbeddings:
    def __init__(self, word_embeddings: WordEmbeddings, prefix_length: int = 3):
        self.word_embeddings = word_embeddings
        self.prefix_length = prefix_length
        self.word_indexer = Indexer()
        self.prefix_vectors = self._initialize_prefix_vectors()

    def _initialize_prefix_vectors(self):
        prefix_to_words = {}
        for word in self.word_embeddings.word_indexer.objs_to_ints:
            prefix = word[:self.prefix_length]
            if prefix not in prefix_to_words:
                prefix_to_words[prefix] = []
            prefix_to_words[prefix].append(word)

        prefix_vectors = []
        for prefix in prefix_to_words:
            self.word_indexer.add_and_get_index(prefix)
            words = prefix_to_words[prefix]
            word_vectors = [self.word_embeddings.get_embedding(word) for word in words]
            avg_vector = sum(word_vectors) / len(word_vectors)
            prefix_vectors.append(avg_vector)

        return torch.tensor(prefix_vectors, dtype=torch.float32)

    def get_initialized_embedding_layer(self, frozen=False):
        return nn.Embedding.from_pretrained(self.prefix_vectors, freeze=frozen)

    def get_embedding_length(self):
        return self.prefix_vectors.shape[1]

    def get_embedding(self, word):
        prefix = word[:self.prefix_length]
        prefix_idx = self.word_indexer.index_of(prefix)
        if prefix_idx != -1:
            return self.prefix_vectors[prefix_idx]
        else:
            return self.prefix_vectors[self.word_indexer.index_of("UNK")]

class NeuralSentimentClassifier(SentimentClassifier):
    def __init__(self, word_embeddings, hidden_size=100, dropout_rate=0.3, use_prefix_embeddings=False):
        if use_prefix_embeddings:
            self.embeddings = PrefixEmbeddings(word_embeddings)
        else:
            self.embeddings = word_embeddings
        self.model = DANSentimentClassifier(self.embeddings, hidden_size, dropout_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, ex_words: List[str], has_typos: bool) -> int:
        if has_typos:
            indexed_words = [self.embeddings.word_indexer.index_of(word[:3]) for word in ex_words]
        else:
            indexed_words = [self.embeddings.word_indexer.index_of(word) for word in ex_words]
        indexed_words = [idx if idx != -1 else self.embeddings.word_indexer.index_of("UNK") for idx in indexed_words]
        tensor = torch.LongTensor(indexed_words).unsqueeze(0).to(self.device)
        log_probs = self.model(tensor)
        return torch.argmax(log_probs, dim=1).item()

    def predict_all(self, all_ex_words: List[List[str]], has_typos: bool) -> List[int]:
        self.model.eval()
        with torch.no_grad():
            if has_typos:
                indexed_words = [[self.embeddings.word_indexer.index_of(word[:3]) if self.embeddings.word_indexer.index_of(word[:3]) != -1 else self.embeddings.word_indexer.index_of("UNK") for word in ex_words] for ex_words in all_ex_words]
            else:
                indexed_words = [[self.embeddings.word_indexer.index_of(word) if self.embeddings.word_indexer.index_of(word) != -1 else self.embeddings.word_indexer.index_of("UNK") for word in ex_words] for ex_words in all_ex_words]
            max_len = max(len(words) for words in indexed_words)
            padded_words = [words + [0] * (max_len - len(words)) for words in indexed_words]
            tensor = torch.LongTensor(padded_words).to(self.device)
            log_probs = self.model(tensor)
            return torch.argmax(log_probs, dim=1).tolist()

def train_deep_averaging_network(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample],
                                 word_embeddings: WordEmbeddings, train_model_for_typo_setting: bool) -> NeuralSentimentClassifier:
    if train_model_for_typo_setting:
        # Use the optimal hyperparameters for typo handling
        hidden_size = 300
        lr = 0.0002
        num_epochs = 50
        batch_size = 64
    else:
        # Use the default or provided hyperparameters for non-typo setting
        hidden_size = args.hidden_size
        lr = args.lr
        num_epochs = args.num_epochs
        batch_size = args.batch_size

    classifier = NeuralSentimentClassifier(word_embeddings, hidden_size, use_prefix_embeddings=train_model_for_typo_setting)
    model = classifier.model
    
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5)
    
    train_data = [(ex.words, ex.label) for ex in train_exs]
    dev_data = [(ex.words, ex.label) for ex in dev_exs]
    
    best_dev_acc = 0
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        random.shuffle(train_data)
        batches = [train_data[i:i + batch_size] for i in range(0, len(train_data), batch_size)]
        
        for batch in batches:
            words, labels = zip(*batch)
            if train_model_for_typo_setting:
                indexed_words = [[classifier.embeddings.word_indexer.index_of(word[:3]) if classifier.embeddings.word_indexer.index_of(word[:3]) != -1 else classifier.embeddings.word_indexer.index_of("UNK") for word in ex_words] for ex_words in words]
            else:
                indexed_words = [[classifier.embeddings.word_indexer.index_of(word) if classifier.embeddings.word_indexer.index_of(word) != -1 else classifier.embeddings.word_indexer.index_of("UNK") for word in ex_words] for ex_words in words]
            max_len = max(len(words) for words in indexed_words)
            padded_words = [words + [0] * (max_len - len(words)) for words in indexed_words]
            
            word_tensor = torch.LongTensor(padded_words).to(classifier.device)
            label_tensor = torch.LongTensor(labels).to(classifier.device)
            
            optimizer.zero_grad()
            log_probs = model(word_tensor)
            loss = criterion(log_probs, label_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        dev_acc = evaluate(classifier, dev_exs, train_model_for_typo_setting)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}, Dev Accuracy: {dev_acc:.4f}")
        
        scheduler.step(dev_acc)
        
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    return classifier

def evaluate(classifier, exs, has_typos):
    predictions = classifier.predict_all([ex.words for ex in exs], has_typos)
    correct = sum(pred == ex.label for pred, ex in zip(predictions, exs))
    return correct / len(exs)
