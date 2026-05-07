import torch
import numpy as np
from rouge_score import rouge_scorer
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class VQAMetrics:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.smooth = SmoothingFunction().method1

    def compute_accuracy(self, preds, labels):
        """Exact match accuracy. Labels can be a list of lists of strings or list of strings."""
        correct = 0
        for p, l in zip(preds, labels):
            p_clean = p.strip().lower()
            if isinstance(l, list):
                l_clean = [ans.strip().lower() for ans in l]
                if p_clean in l_clean:
                    correct += 1
            elif p_clean == l.strip().lower():
                correct += 1
        return correct / len(preds) if preds else 0

    def compute_batch_metrics(self, preds, gts):
        """
        preds: list of strings
        gts: list of strings OR list of list of strings (multiple refs)
        """
        acc = self.compute_accuracy(preds, gts)
        
        bleu_scores = []
        rouge_scores = []
        
        for p, g in zip(preds, gts):
            # Ensure g is a list of references
            refs = g if isinstance(g, list) else [g]
            
            # BLEU: Take max BLEU across all references
            p_split = p.split()
            b_score = max([sentence_bleu([r.split()], p_split, smoothing_function=self.smooth) for r in refs])
            bleu_scores.append(b_score)
            
            # ROUGE: Take max ROUGE across all references
            r_score = max([self.rouge_scorer.score(r, p)['rougeL'].fmeasure for r in refs])
            rouge_scores.append(r_score)
            
        return {
            'accuracy': acc,
            'bleu': np.mean(bleu_scores),
            'rougeL': np.mean(rouge_scores),
        }
