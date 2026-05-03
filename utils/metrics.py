import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk

# Ensure nltk resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class VQAMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smoothie = SmoothingFunction().method1

    def decode_ids(self, ids):
        """Convert token IDs back to string, skipping special tokens."""
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip().lower()

    def compute_accuracy(self, preds, targets):
        """Calculate Exact Match Accuracy."""
        correct = 0
        for p, t in zip(preds, targets):
            if p == t:
                correct += 1
        return correct / len(preds) if len(preds) > 0 else 0

    def compute_batch_metrics(self, logits, target_ids):
        """
        Calculate metrics for a batch of predictions.
        logits: [batch, seq_len, vocab_size]
        target_ids: [batch, seq_len]
        """
        preds_ids = torch.argmax(logits, dim=-1)
        
        batch_preds = [self.decode_ids(p) for p in preds_ids]
        batch_targets = [self.decode_ids(t) for t in target_ids]
        
        # 1. Accuracy (Exact Match)
        acc = self.compute_accuracy(batch_preds, batch_targets)
        
        # 2. BLEU & ROUGE-L
        bleu_scores = []
        rouge_scores = []
        
        for p, t in zip(batch_preds, batch_targets):
            # BLEU-4
            p_tokens = p.split()
            t_tokens = t.split()
            if not t_tokens: # Handle empty target
                bleu = 0.0
                rouge = 0.0
            else:
                bleu = sentence_bleu([t_tokens], p_tokens, smoothing_function=self.smoothie)
                rouge = self.rouge_scorer.score(t, p)['rougeL'].fmeasure
            
            bleu_scores.append(bleu)
            rouge_scores.append(rouge)
            
        return {
            'accuracy': acc,
            'bleu': np.mean(bleu_scores),
            'rougeL': np.mean(rouge_scores),
            'preds_sample': batch_preds[:2],   # Return samples for logging
            'targets_sample': batch_targets[:2]
        }
