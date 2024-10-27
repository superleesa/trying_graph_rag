import ollama
import networkx as nx

from sentence_transformers import SentenceTransformer

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

# Define function to get relationship using the generator model
def get_relationship(doc1_text, doc2_text):
    prompt = f"Determine if there is a relationship between the following two texts, and if so, describe why:\n\nText 1: {doc1_text}\n\nText 2: {doc2_text}\n\nIf they are related, explain the relationship briefly. If they are unrelated, just respond with 'No meaningful connection.'"
    
    # Tokenize prompt and generate response
    ollama_response = ollama.chat(model='gemma2:2b', messages=[
        {
            'role': 'user',
            'content': prompt,
        },
    ])
    return ollama_response['message']['content']

# Generate relationships using the generator model
for i, (doc_id1, text1) in enumerate(documents.items()):
    for j, (doc_id2, text2) in enumerate(documents.items()):
        if i < j:  # Avoid duplicate pairs and self-loops
            relationship = get_relationship(text1, text2)
            print(relationship)
            if relationship != "No meaningful connection.":
                graph.add_edge(doc_id1, doc_id2, description=relationship)

# Adding nodes and their content as attributes
for doc_id, text in documents.items():
    graph.add_node(doc_id, text=text)

# Display edges with their LLM-generated relationships
for edge in graph.edges(data=True):
    print(edge)
