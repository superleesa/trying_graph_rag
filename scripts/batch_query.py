import json
import pickle
from pathlib import Path

import fire

from trying_graph_rag.graph_rag.querying import query_index

DEFAULT_ENTITY_TYPES = ["PERSON", "ORGANIZATION", "LOCATION", "TECH", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "ANIMAL", "MISSION"]

def main(test_cases_json_path: str, indexes_base_path: str, output_answers_csv_path: str) -> None:
    with open(test_cases_json_path, "r") as file:
        test_cases = json.load(file)

    _ouput_answeres_csv_path = Path(output_answers_csv_path)
    with open(_ouput_answeres_csv_path, "w") as file:
        file.write("document_id,answer,explanation\n")
    
    for test_case in test_cases:
        document_id, question = test_case["_id"], test_case["question"]
        
        with open(Path(indexes_base_path) / f"{document_id}.pickle", "rb") as file:
            index = pickle.load(file)
        
        answer, explanation = query_index(question.strip(), index)

        with open(_ouput_answeres_csv_path, "a") as file:
            file.write(f"{document_id},{answer},{explanation}\n")
        

if __name__ == '__main__':
    fire.Fire(main)