import re

# Simple tokenizer function
def simple_tokenize(text):
    # Convert to lowercase and split based on non-word characters
    return re.findall(r'\b\w+\b', text.lower())

# Define exact match metric
def exact_match(prediction, ground_truth):
    return int(prediction.strip().lower() == ground_truth.strip().lower())

# Define F1 score calculation without nltk
def calculate_f1(prediction, ground_truth):
    prediction_tokens = set(simple_tokenize(prediction))
    ground_truth_tokens = set(simple_tokenize(ground_truth))
    common_tokens = prediction_tokens & ground_truth_tokens
    
    if not common_tokens:
        return 0.0
    
    precision = len(common_tokens) / len(prediction_tokens)
    recall = len(common_tokens) / len(ground_truth_tokens)
    return 2 * (precision * recall) / (precision + recall)

# Calculate metrics for a batch of predictions
def calculate_metrics(predictions, references):
    em_scores = []
    f1_scores = []
    
    for pred, ref in zip(predictions, references):
        em_scores.append(exact_match(pred, ref))
        f1_scores.append(calculate_f1(pred, ref))
    
    # Average scores
    em = sum(em_scores) / len(em_scores)
    f1 = sum(f1_scores) / len(f1_scores)
    return em, f1