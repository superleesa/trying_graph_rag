import json

import fire
from tqdm import tqdm

from trying_graph_rag.graph_rag.indexing import create_index

DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "TECH", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "ANIMAL", "MISSION"]

def main(test_cases_json_path: str) -> None:
    with open(test_cases_json_path, "r") as file:
        test_cases = json.load(file)
    
    for test_case in tqdm(test_cases, desc="Test cases"):
        document_id, documents = test_case["_id"], test_case["context"]
        create_index(documents, DEFAULT_ENTITY_TYPES, index_name=document_id)

if __name__ == '__main__':
    fire.Fire(main)