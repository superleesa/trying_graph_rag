import json

import fire

from trying_graph_rag.graph_rag.indexing import create_index

DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "TECH", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "ANIMAL", "MISSION"]

def main(test_cases_json_path: str) -> None:
    with open(test_cases_json_path, "r") as file:
        test_cases = json.load(file)
    
    for document in test_cases:
        document_id, test_cases = document["_id"], test_cases["context"]
        create_index(test_cases, DEFAULT_ENTITY_TYPES, index_name=document_id)

if __name__ == '__main__':
    fire.Fire(main)