# factcheck.py
import torch
from typing import List, Tuple
import numpy as np
import spacy
import gc
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from nltk.corpus import stopwords
import string
import re

# Load the English stopwords
stop_words = set(stopwords.words('english'))

class FactExample:
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))

class EntailmentModel:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
            entailment_prob = probs[0]
            contradiction_prob = probs[2]
            return entailment_prob, contradiction_prob

        del inputs, outputs, logits
        gc.collect()

class WordRecallThresholdFactChecker:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        
    def normalize_text(self, text):
        # More aggressive normalization
        text = text.lower()
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'\d+', '', text)  # Remove numbers
        text = text.translate(str.maketrans('', '', string.punctuation.replace('-', '')))
        text = ' '.join(text.split())
        return text
        
    def get_words(self, text):
        text = self.normalize_text(text)
        # Enhanced word processing
        words = []
        for w in text.split():
            if w not in self.stop_words and len(w) > 2:
                # More aggressive suffix stripping
                if w.endswith('ing'):
                    w = w[:-3]
                elif w.endswith('ed'):
                    w = w[:-2]
                elif w.endswith('s'):
                    w = w[:-1]
                elif w.endswith('ly'):
                    w = w[:-2]
                elif w.endswith('ment'):
                    w = w[:-4]
                words.append(w)
        return words

    def get_ratio_score(self, matches, total, min_matches=1):
        if matches < min_matches:
            return 0.0
        return matches / total if total > 0 else 0.0

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_words = set(self.get_words(fact))
        if not fact_words:
            return "NS"
            
        # Enhanced tracking
        best_recall = 0.0
        best_precision = 0.0
        meaningful_matches = 0
        fact_length = len(fact.split())
        
        for passage in passages:
            passage_words = set(self.get_words(passage["text"]))
            if not passage_words:
                continue
                
            overlap = fact_words & passage_words
            
            # Dynamic minimum matches based on fact length
            min_matches = max(2, min(len(fact_words) - 1, 4))
            
            recall = self.get_ratio_score(len(overlap), len(fact_words), min_matches)
            precision = self.get_ratio_score(len(overlap), min(len(passage_words), 30))
            
            if len(overlap) >= min_matches:
                meaningful_matches += 1
            
            best_recall = max(best_recall, recall)
            best_precision = max(best_precision, precision)
            
            # Adjusted early exit conditions
            if recall > 0.9 and precision > 0.4:
                return "S"
            if recall > 0.8 and precision > 0.5 and meaningful_matches >= 2:
                return "S"
        
        # Dynamic scoring based on fact characteristics
        weight_recall = 0.75 if fact_length <= 5 else 0.65
        weight_precision = 1 - weight_recall
        score = (weight_recall * best_recall) + (weight_precision * best_precision)
        
        # Dynamic threshold
        threshold = 0.4
        if fact_length <= 4:
            threshold = 0.45
        elif meaningful_matches >= 2:
            threshold = 0.35
        
        return "S" if score > threshold else "NS"

class EntailmentFactChecker:
    def __init__(self, ent_model):
        self.ent_model = ent_model
        self.word_checker = WordRecallThresholdFactChecker()
        self.model = ent_model.model
        self.tokenizer = ent_model.tokenizer

    def get_scores(self, premise, hypothesis):
        with torch.no_grad():
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            
            del inputs, outputs
            gc.collect()
            
            return probs[0], probs[1], probs[2]

    def check_logical_implication(self, fact: str, passage: str) -> float:
        """Check if the fact is logically implied by the passage"""
        fact_words = fact.lower().split()
        passage = passage.lower()
        
        # Identity facts and property implications
        is_identity = ('is' in fact_words or 'was' in fact_words) and len(fact_words) <= 6
        if is_identity:
            words = [w for w in fact_words if w not in {'is', 'was', 'a', 'an', 'the'}]
            if len(words) <= 3:
                # Extended property implications
                property_implications = {
                    'writer': ['novelist', 'author', 'poet', 'journalist', 'columnist'],
                    'performer': ['actor', 'actress', 'singer', 'musician', 'dancer', 'artist'],
                    'artist': ['painter', 'sculptor', 'illustrator', 'designer', 'creator'],
                    'leader': ['president', 'chairman', 'director', 'chief', 'head', 'founder'],
                    'public figure': ['politician', 'activist', 'advocate', 'representative'],
                    'professional': ['expert', 'specialist', 'authority', 'practitioner']
                }
                
                for prop in words:
                    if prop in property_implications:
                        if any(stronger in passage for stronger in property_implications[prop]):
                            return 0.95

        # Location implications
        if 'in' in fact_words and len(fact_words) <= 6:
            location_pairs = {
                'california': 'usa',
                'new york': 'usa',
                'paris': 'france',
                'london': 'england',
                'tokyo': 'japan'
            }
            for loc1, loc2 in location_pairs.items():
                if loc1 in passage.lower() and loc2 in fact.lower():
                    return 0.9
                    
        # Time expressions with enhanced handling
        date_indicators = ['since', 'from', 'until', 'between', 'in', 'on']
        if any(ind in fact.lower() for ind in date_indicators):
            fact_dates = re.findall(r'\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}', fact)
            passage_dates = re.findall(r'\d{4}|\d{1,2}/\d{1,2}/\d{4}|\d{1,2}-\d{1,2}-\d{4}', passage)
            if fact_dates and fact_dates[0] in passage_dates:
                return 0.95
                    
        return 0.0

    def clean_text(self, text):
        """Enhanced text cleaning"""
        # Remove citations and HTML
        text = re.sub(r'\[\d+\]|<.*?>', '', text)
        # Standardize quotes and apostrophes
        text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        # Standardize different forms
        replacements = {
            'known as': 'also called',
            'referred to as': 'also called',
            'nicknamed': 'also called',
            'born in': 'from',
            'native of': 'from',
            'originally from': 'from'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text.strip()

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact = fact.strip()
        has_number = any(c.isdigit() for c in fact)
        fact_words = fact.lower().split()
        
        # Quick check for logical implications with enhanced confidence
        for passage in passages[:3]:  # Check one more passage
            logical_score = self.check_logical_implication(fact, passage["text"])
            if logical_score > 0.9:  # Increased threshold
                ent_score, _, contra_score = self.get_scores(passage["text"], fact)
                if contra_score < 0.2:  # More lenient contradiction threshold
                    return "S"

        # Smarter word overlap check
        max_overlap = 0
        significant_matches = 0
        for p in passages[:3]:  # Check one more passage
            words1 = set(self.word_checker.get_words(fact))
            words2 = set(self.word_checker.get_words(p["text"]))
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1)
                max_overlap = max(max_overlap, overlap)
                if overlap > 0.4:  # Track significant matches
                    significant_matches += 1
                
        # Early return for very low overlap, but more lenient
        if max_overlap < 0.1 and not has_number:  # Even more lenient for non-numerical facts
            return "NS"

        # Track different types of evidence
        best_entail_score = 0.0
        strong_evidence = 0
        moderate_evidence = 0
        weak_evidence = 0
        min_contradiction = 1.0
        total_chunks_checked = 0

        for passage in passages:
            text = self.clean_text(passage["text"])
            
            # Improved chunking strategy
            chunks = []
            sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]
            current_chunk = ""
            
            for sentence in sentences:
                if len(current_chunk) + len(sentence) < 450:  # Slightly smaller chunks
                    current_chunk += " " + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
            if current_chunk:
                chunks.append(current_chunk)
            
            # Check each chunk
            for chunk in chunks:
                total_chunks_checked += 1
                ent_score, neutral_score, contra_score = self.get_scores(chunk, fact)
                
                min_contradiction = min(min_contradiction, contra_score)
                best_entail_score = max(best_entail_score, ent_score)
                
                # Adjusted thresholds
                if ent_score > 0.85 and contra_score < 0.2:
                    strong_evidence += 1
                elif ent_score > 0.7 and contra_score < 0.3:
                    moderate_evidence += 1
                elif ent_score > 0.5 and contra_score < 0.4:
                    weak_evidence += 1
                    
                # Quick positive decisions with high confidence
                if ent_score > 0.9 and contra_score < 0.1:
                    return "S"
                    
                if strong_evidence >= 2 or (strong_evidence == 1 and moderate_evidence >= 2):
                    return "S"

                # Memory management
                if total_chunks_checked % 10 == 0:
                    gc.collect()

        # Final decision logic
        if has_number:
            # Numerical facts require stronger evidence
            if best_entail_score > 0.85 and min_contradiction < 0.15:
                return "S"
            if strong_evidence > 0 and significant_matches >= 1:
                return "S"
            return "NS"
            
        # More nuanced thresholds for non-numerical facts
        if strong_evidence > 0:
            if moderate_evidence > 0 or max_overlap > 0.35:
                return "S"
                
        if moderate_evidence >= 2 and max_overlap > 0.25:
            return "S"
            
        if best_entail_score > 0.75 and max_overlap > 0.35 and min_contradiction < 0.25:
            return "S"
        
        # Special case for simple identity claims
        if len(fact_words) <= 5 and ('is' in fact_words or 'was' in fact_words):
            if best_entail_score > 0.7 and max_overlap > 0.5 and min_contradiction < 0.3:
                return "S"
            
        return "NS"
        
class DependencyRecallThresholdFactChecker:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        fact_dependencies = self.get_dependencies(fact)
        max_score = 0
        for passage in passages:
            sentences = passage["text"].split(".")
            for sentence in sentences:
                passage_dependencies = self.get_dependencies(sentence)
                overlap_score = len(fact_dependencies & passage_dependencies) / len(fact_dependencies)
                max_score = max(max_score, overlap_score)
        threshold = 0.4
        return "S" if max_score >= threshold else "NS"

    def get_dependencies(self, sent: str):
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            head = token.head.text.lower()
            dependent = token.text.lower()
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations
