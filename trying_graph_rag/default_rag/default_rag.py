from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import ollama

# Load the dense retriever
retriever = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

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

def generate_answer_ollama(query, retrieved_docs):
    input_text = f"{query} Context: {retrieved_docs}"
    response = ollama.generate(model="gemma2:2b", prompt=input_text)  # Specify model name here if different in Ollama
    generated_answer = response['response']
    return generated_answer

# Example usage
query = "How does RAG work?"
retrieved_docs = retrieve_documents(query)
generated_answer = generate_answer_ollama(query, retrieved_docs)

print("Query:", query)
print("Retrieved Documents:", retrieved_docs)
print("Generated Answer:", generated_answer)