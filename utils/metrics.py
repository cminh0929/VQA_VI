import nltk
from nltk.translate.bleu_score import sentence_bleu
from evaluate import load
from bert_score import score

# Pre-load metrics
rouge = load("rouge")
# meteor = load("meteor") # Requires java, optional

def calculate_vqa_accuracy(predictions, ground_truths):
    correct = 0
    for pred, gt in zip(predictions, ground_truths):
        if pred.lower().strip() == gt.lower().strip():
            correct += 1
    return correct / len(predictions) if predictions else 0

def calculate_metrics(predictions, ground_truths):
    # BLEU
    bleu_scores = []
    for pred, gt in zip(predictions, ground_truths):
        bleu_scores.append(sentence_bleu([gt.split()], pred.split()))
    avg_bleu = sum(bleu_scores) / len(bleu_scores) if bleu_scores else 0
    
    # ROUGE
    rouge_results = rouge.compute(predictions=predictions, references=ground_truths)
    
    # BERTScore
    P, R, F1 = score(predictions, ground_truths, lang="vi", verbose=False)
    avg_bert_f1 = F1.mean().item()
    
    return {
        "vqa_accuracy": calculate_vqa_accuracy(predictions, ground_truths),
        "bleu": avg_bleu,
        "rougeL": rouge_results['rougeL'],
        "bert_f1": avg_bert_f1
    }
