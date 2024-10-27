import networkx as nx
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Initialize graph and documents
graph = nx.Graph()
documents = {
    "doc1": "Python is a popular programming language.",
    "doc2": "RAG combines retrieval with generation to produce more accurate answers.",
    "doc3": "Transformers are deep learning models used for NLP tasks.",
    "doc4": "PyTorch is a deep learning library often used with transformers.",
    # Add more documents or load from a larger corpus
}

# Initialize dense retriever for initial retrieval and embeddings
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Encode document embeddings
doc_texts = list(documents.values())
document_embeddings = retriever.encode(doc_texts, convert_to_tensor=True)

# Initialize the generator model and tokenizer
generator_model = "facebook/bart-large-cnn"
generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
tokenizer = AutoTokenizer.from_pretrained(generator_model)

# Define function to get relationship using the generator model
def get_relationship(doc1_text, doc2_text):
    prompt = f"Determine if there is a relationship between the following two texts, and if so, describe it:\n\nText 1: {doc1_text}\n\nText 2: {doc2_text}\n\nIf they are related, explain the relationship briefly. If they are unrelated, respond with 'No meaningful connection.'"
    
    # Tokenize prompt and generate response
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = generator.generate(inputs["input_ids"], max_length=200, min_length=1, length_penalty=2.0, num_beams=4, early_stopping=True)
    response = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    print(response)
    return response

# Generate relationships using the generator model
for i, (doc_id1, text1) in enumerate(documents.items()):
    for j, (doc_id2, text2) in enumerate(documents.items()):
        if i < j:  # Avoid duplicate pairs and self-loops
            relationship = get_relationship(text1, text2)
            if relationship != "No meaningful connection.":
                graph.add_edge(doc_id1, doc_id2, description=relationship)

# Adding nodes and their content as attributes
for doc_id, text in documents.items():
    graph.add_node(doc_id, text=text)

# Display edges with their LLM-generated relationships
for edge in graph.edges(data=True):
    print(edge)