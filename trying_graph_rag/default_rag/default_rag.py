from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import faiss
import ollama
import json
import re
import os

# Load the dense retriever
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def encode_context(context):
    """Encodes the context sentences for each question individually."""
    context_sentences = []
    for title, sentences in context:
        for sentence in sentences:
            context_sentences.append((title, sentence))
    return context_sentences

def create_faiss_index(sentences):
    """Creates a FAISS index for the provided sentences."""
    embeddings = retriever.encode([s[1] for s in sentences], convert_to_tensor=True)
    d = embeddings.shape[1]  # Dimensionality of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings.cpu().detach().numpy())
    return index, embeddings

def retrieve_documents(query, context_sentences, k=3):
    """Retrieves relevant sentences from the dynamic context."""
    query_embedding = retriever.encode([query], convert_to_tensor=True)
    index, embeddings = create_faiss_index(context_sentences)
    _, indices = index.search(query_embedding.cpu().detach().numpy(), k)
    retrieved_docs = [context_sentences[idx][1] for idx in indices[0]]
    return " ".join(retrieved_docs)  # Concatenate sentences

def generate_answer_ollama(query, retrieved_docs):
    """Generates an answer using Ollama's model."""
    input_text = f"{query} Context: {retrieved_docs}"
    response = ollama.generate(model="gemma2:2b", prompt=input_text)  # Specify model name here if different in Ollama
    generated_answer = response['response']
    return generated_answer

# Define the relevance function
def model_decides_sentence_is_relevant(sentence, answer):
    # Simple relevance model: check for keyword overlap
    sentence_words = set(re.findall(r'\w+', sentence.lower()))
    answer_words = set(re.findall(r'\w+', answer.lower()))
    common_words = sentence_words & answer_words
    return len(common_words) > 0  # Adjust relevance threshold as needed

# Load test data
with open('/Users/sho/Monash/FIT5047/project/chunking_strategies/sampled_gold_data.json', 'r') as f:
    test_data = json.load(f)

# Check if the prediction file exists, if so, load its content; otherwise, initialize empty dictionaries
pred_file = "dev_distractor_pred.json"
if os.path.exists(pred_file):
    with open(pred_file, "r") as f:
        predictions = json.load(f)
else:
    predictions = {"answer": {}, "sp": {}}

for item in tqdm(test_data):
    question_id = item["_id"]
    if question_id in predictions["answer"]:  # Skip already processed questions
        continue
    
    question_text = item["question"]
    context = item["context"]

    # Encode and retrieve documents based on context
    context_sentences = encode_context(context)
    retrieved_docs = retrieve_documents(question_text, context_sentences)

    # Generate answer based on retrieved context
    answer = generate_answer_ollama(question_text, retrieved_docs)

    # Identify supporting facts based on relevance
    supporting_facts = []
    for title, sentences in context:
        for sent_id, sentence in enumerate(sentences):
            if model_decides_sentence_is_relevant(sentence, answer):
                supporting_facts.append([title, sent_id])

    # Store predictions
    predictions["answer"][question_id] = answer
    predictions["sp"][question_id] = supporting_facts

    # Save to file incrementally
    with open(pred_file, "w") as f:
        json.dump(predictions, f, indent=4)

    print(f"Saved prediction for question ID: {question_id}")

print("All predictions saved to dev_distractor_pred.json")
