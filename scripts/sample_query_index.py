import pickle

from trying_graph_rag.graph_rag.querying import query_index

SAMPLE_INDEX_PATH = "index.pickle"
SAMPLE_QUERY = "Who did Sofia Martinez collaborate with?"


if __name__ == '__main__':
    
    with open(SAMPLE_INDEX_PATH, "rb") as file:
        sample_index = pickle.load(file)

    print(query_index(SAMPLE_QUERY, sample_index))
