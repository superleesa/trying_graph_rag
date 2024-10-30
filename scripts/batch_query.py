import json
import pickle
from pathlib import Path

import fire
from tqdm import tqdm

from trying_graph_rag.graph_rag.querying import query_index


def main(test_cases_json_path: str, indexes_base_path: str, output_answers_csv_path: str) -> None:
    with open(test_cases_json_path, "r") as file:
        test_cases = json.load(file)

    _ouput_answeres_csv_path = Path(output_answers_csv_path)
    with open(_ouput_answeres_csv_path, "w") as file:
        file.write("document_id,answer,explanation\n")
    
    for test_case in tqdm(test_cases, desc="Test cases"):
        document_id, question = test_case["_id"], test_case["question"]
        
        try:
            with open(Path(indexes_base_path) / f"{document_id}.pickle", "rb") as file:
                index = pickle.load(file)
        except FileNotFoundError:
            print(f"Index for document {document_id} not found, skipping...")
            continue
        
        answer, explanation = query_index(question.strip(), index)

        with open(_ouput_answeres_csv_path, "a") as file:
            file.write(f"{document_id},{answer},{explanation}\n")
        

if __name__ == '__main__':
    fire.Fire(main)