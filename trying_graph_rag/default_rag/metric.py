import re
import json
from tqdm import tqdm

from default_rag import retrieve_documents, generate_answer_ollama

# Simple tokenizer for calculating F1 without nltk
def simple_tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

# Define metrics
def exact_match(prediction, ground_truth):
    return int(prediction.strip().lower() == ground_truth.strip().lower())

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

with open("/Users/sho/Monash/FIT5047/project/chunking_strategies/sampled_test_data.json", "r") as file:
    test_data = json.load(file)

# Run the RAG model and evaluate
predictions = []
references = []

for item in tqdm(test_data):
    query = item["question"]
    true_answer = item["true_answer"]
    
    # Retrieve and generate answer
    retrieved_docs = retrieve_documents(query)
    generated_answer = generate_answer_ollama(query, retrieved_docs)
    
    predictions.append(generated_answer)
    references.append(true_answer)
    
    print(f"Query: {query}")
    print(f"Retrieved Documents: {retrieved_docs}")
    print(f"Generated Answer: {generated_answer}")
    print(f"True Answer: {true_answer}\n")

# Calculate metrics
em, f1 = calculate_metrics(predictions, references)
print(f"Overall Exact Match: {em * 100:.2f}%")
print(f"Overall F1 Score: {f1 * 100:.2f}%")
