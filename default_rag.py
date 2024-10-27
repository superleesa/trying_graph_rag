import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss

# Load the dense retriever and the generative model
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
generator_model = "facebook/bart-large-cnn"
generator = AutoModelForSeq2SeqLM.from_pretrained(generator_model)
tokenizer = AutoTokenizer.from_pretrained(generator_model)

# Define a small corpus of documents for retrieval
documents = [
    "Python is a popular programming language.",
    "RAG combines retrieval with generation to produce more accurate answers.",
    "Transformers are deep learning models used for NLP tasks.",
    "PyTorch is a deep learning library often used with transformers.",
    # Add more documents as needed
]

# Encode the documents into embeddings
document_embeddings = retriever.encode(documents, convert_to_tensor=True)

# Create an FAISS index for similarity search
d = document_embeddings.shape[1]  # Dimensionality of embeddings
index = faiss.IndexFlatL2(d)
index.add(document_embeddings.cpu().detach().numpy())  # Adding to the index

def retrieve_documents(query, k=3):
    query_embedding = retriever.encode([query], convert_to_tensor=True)
    _, indices = index.search(query_embedding.cpu().detach().numpy(), k)
    retrieved_docs = [documents[idx] for idx in indices[0]]
    return " ".join(retrieved_docs)

def generate_answer(query, retrieved_docs):
    input_text = f"{query} Context: {retrieved_docs}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = generator.generate(inputs["input_ids"], max_length=150, min_length=30, length_penalty=2.0, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# Example usage
query = "How does RAG work?"
retrieved_docs = retrieve_documents(query)
generated_answer = generate_answer(query, retrieved_docs)

print("Query:", query)
print("Retrieved Documents:", retrieved_docs)
print("Generated Answer:", generated_answer)