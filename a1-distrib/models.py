# models.py

from sentiment_data import *
from utils import *
import numpy as np
import random
from collections import Counter
from typing import List

# A small set of stopwords
stopwords = set(["the", "and", "a", "an", "of", "to", "in", "that", "it", "is", "on", "with", "as", "this", "for"])

class FeatureExtractor(object):
    """
    Feature extraction base type. Takes a sentence and returns an indexed list of features.
    """
    def get_indexer(self):
        raise Exception("Don't call me, call my subclasses")

    def extract_features(self, sentence: List[str], add_to_indexer: bool=False) -> Counter:
        raise Exception("Don't call me, call my subclasses")

class UnigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        for word in sentence:
            if word not in stopwords:  # Skip stopwords
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(word, add=True)
                else:
                    idx = self.indexer.index_of(word)
                    if idx == -1:
                        continue
                features[idx] += 1
        return features

class BigramFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        # Unigram features
        for word in sentence:
            if word not in stopwords:  # Skip stopwords
                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"Unigram={word}", add=True)
                else:
                    idx = self.indexer.index_of(f"Unigram={word}")
                    if idx == -1:
                        continue
                features[idx] += 1
        
        # Bigram features
        for i in range(len(sentence) - 1):
            bigram = f"{sentence[i]} {sentence[i + 1]}"
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"Bigram={bigram}", add=True)
            else:
                idx = self.indexer.index_of(f"Bigram={bigram}")
                if idx == -1:
                    continue
            features[idx] += 1
        return features

class BetterFeatureExtractor(FeatureExtractor):
    def __init__(self, indexer: Indexer):
        self.indexer = indexer
        self.doc_freqs = Counter()
        self.total_docs = 0

    def get_indexer(self):
        return self.indexer

    def extract_features(self, sentence: List[str], add_to_indexer: bool = False) -> Counter:
        features = Counter()
        sentence_length = len(sentence)
        words_in_sentence = set(sentence)

        # Update document frequencies (only in training)
        if add_to_indexer:
            self.total_docs += 1
            for word in words_in_sentence:
                self.doc_freqs[word] += 1

        # Unigram features with TF-IDF and stopword removal
        for word in sentence:
            if word not in stopwords:  # Skip stopwords
                tf = sentence.count(word) / sentence_length  # Term Frequency (TF)
                df = self.doc_freqs.get(word, 1)  # Document Frequency (DF)
                idf = np.log(self.total_docs / df)  # Inverse Document Frequency (IDF)
                tfidf = tf * idf  # TF-IDF score

                if add_to_indexer:
                    idx = self.indexer.add_and_get_index(f"Unigram={word}", add=True)
                else:
                    idx = self.indexer.index_of(f"Unigram={word}")
                    if idx == -1:
                        continue
                features[idx] += tfidf

        # Bigram features
        for i in range(len(sentence) - 1):
            bigram = f"{sentence[i]} {sentence[i + 1]}"
            if add_to_indexer:
                idx = self.indexer.add_and_get_index(f"Bigram={bigram}", add=True)
            else:
                idx = self.indexer.index_of(f"Bigram={bigram}")
                if idx == -1:
                    continue
            features[idx] += 1

        # Add sentiment lexicon features
        positive_words = set(["good", "great", "excellent", "amazing", "wonderful", "best", "love", "favorite"])
        negative_words = set(["bad", "terrible", "awful", "horrible", "worst", "hate", "disappointing", "boring"])
        
        pos_count = sum(1 for word in sentence if word.lower() in positive_words)
        neg_count = sum(1 for word in sentence if word.lower() in negative_words)
        
        if add_to_indexer:
            pos_idx = self.indexer.add_and_get_index("POSITIVE_WORDS", add=True)
            neg_idx = self.indexer.add_and_get_index("NEGATIVE_WORDS", add=True)
        else:
            pos_idx = self.indexer.index_of("POSITIVE_WORDS")
            neg_idx = self.indexer.index_of("NEGATIVE_WORDS")
        
        if pos_idx != -1:
            features[pos_idx] = pos_count / sentence_length
        if neg_idx != -1:
            features[neg_idx] = neg_count / sentence_length

        # Add sentence length feature
        if add_to_indexer:
            len_idx = self.indexer.add_and_get_index("SENTENCE_LENGTH", add=True)
        else:
            len_idx = self.indexer.index_of("SENTENCE_LENGTH")
        if len_idx != -1:
            features[len_idx] = sentence_length / 100  # Normalize by dividing by 100

        # Add negation feature
        negation_words = set(["not", "no", "never", "neither", "nor", "hardly", "scarcely"])
        has_negation = any(word.lower() in negation_words for word in sentence)
        if add_to_indexer:
            neg_idx = self.indexer.add_and_get_index("HAS_NEGATION", add=True)
        else:
            neg_idx = self.indexer.index_of("HAS_NEGATION")
        if neg_idx != -1:
            features[neg_idx] = 1 if has_negation else 0

        return features

class SentimentClassifier(object):
    def predict(self, sentence: List[str]) -> int:
        raise Exception("Don't call me, call my subclasses")

class TrivialSentimentClassifier(SentimentClassifier):
    def predict(self, sentence: List[str]) -> int:
        return 1

class PerceptronClassifier(SentimentClassifier):
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[idx] * count for idx, count in features.items())
        return 1 if score >= 0 else 0

class LogisticRegressionClassifier(SentimentClassifier):
    def __init__(self, weights: np.ndarray, feat_extractor: FeatureExtractor):
        self.weights = weights
        self.feat_extractor = feat_extractor

    def predict(self, sentence: List[str]) -> int:
        features = self.feat_extractor.extract_features(sentence)
        score = sum(self.weights[idx] * count for idx, count in features.items())
        prob = 1 / (1 + np.exp(-score))  # Sigmoid function
        return 1 if prob >= 0.5 else 0

def train_perceptron(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor) -> PerceptronClassifier:
    num_epochs = 12  # Increased from 10 to 12
    learning_rate = 0.05  # Adjusted from 0.1 to 0.05

    # Ensure features are extracted and indexed before determining vocabulary size
    for example in train_exs:
        feat_extractor.extract_features(example.words, add_to_indexer=True)

    # Now initialize the weight vector based on the size of the feature indexer
    vocab_size = len(feat_extractor.get_indexer())
    weights = np.zeros(vocab_size)

    # Training loop
    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for example in train_exs:
            features = feat_extractor.extract_features(example.words, add_to_indexer=False)
            prediction = sum(weights[idx] * count for idx, count in features.items())
            label = 1 if prediction >= 0 else 0
            if label != example.label:
                for idx, count in features.items():
                    weights[idx] += learning_rate * (example.label - label) * count

    return PerceptronClassifier(weights, feat_extractor)

def train_logistic_regression(train_exs: List[SentimentExample], feat_extractor: FeatureExtractor, l2_lambda=0.0005) -> LogisticRegressionClassifier:
    num_epochs = 25  # Increased from 20 to 25
    learning_rate = 0.003  # Adjusted from 0.005 to 0.003

    # Ensure features are extracted and indexed before determining vocabulary size
    for example in train_exs:
        feat_extractor.extract_features(example.words, add_to_indexer=True)

    # Initialize the weight vector based on the size of the feature indexer
    vocab_size = len(feat_extractor.get_indexer())
    weights = np.zeros(vocab_size)

    # Training loop with L2 regularization
    for epoch in range(num_epochs):
        random.shuffle(train_exs)
        for example in train_exs:
            features = feat_extractor.extract_features(example.words, add_to_indexer=False)
            score = sum(weights[idx] * count for idx, count in features.items())
            prob = 1 / (1 + np.exp(-score))  # Sigmoid function
            error = example.label - prob
            for idx, count in features.items():
                weights[idx] += learning_rate * (error * count - l2_lambda * weights[idx])  # L2 regularization

    return LogisticRegressionClassifier(weights, feat_extractor)

def train_model(args, train_exs: List[SentimentExample], dev_exs: List[SentimentExample]) -> SentimentClassifier:
    if args.model == "TRIVIAL":
        feat_extractor = None
    elif args.feats == "UNIGRAM":
        feat_extractor = UnigramFeatureExtractor(Indexer())
    elif args.feats == "BIGRAM":
        feat_extractor = BigramFeatureExtractor(Indexer())
    elif args.feats == "BETTER":
        feat_extractor = BetterFeatureExtractor(Indexer())
    else:
        raise Exception("Pass in UNIGRAM, BIGRAM, or BETTER to run the appropriate system")

    if args.model == "TRIVIAL":
        model = TrivialSentimentClassifier()
    elif args.model == "PERCEPTRON":
        model = train_perceptron(train_exs, feat_extractor)
    elif args.model == "LR":
        model = train_logistic_regression(train_exs, feat_extractor, l2_lambda=0.0005)  # Adjusted regularization parameter
    else:
        raise Exception("Pass in TRIVIAL, PERCEPTRON, or LR to run the appropriate system")
    return model
