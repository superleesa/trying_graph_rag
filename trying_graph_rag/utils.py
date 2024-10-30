import ollama
from ollama import Options
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
OLLAMA_GENERATION_CONFIG = Options(temperature=0, seed=123456789, top_k=1, repeat_penalty=1.2)

def generate_ollama_response(prompt: str) -> str:
    return ollama.generate(
        model="gemma2:2b",
        prompt=prompt,
        options=OLLAMA_GENERATION_CONFIG,
    )["response"]


def flatten[T](tuple_list: tuple[list[T]]) -> list[T]:
    return [item for sublist in tuple_list for item in sublist]


def filter_non_fittable_elements(elements: list[str], max_length: int, element_delimiter: str) -> list[str]:
    """
    when assigning `max_length` consider the output length and the default prompt length
    """
    delimiter_token_length = len(tokenizer(element_delimiter)["input_ids"])
    filtered_elements = []
    current_length = 0
    for element in elements:
        element_token_length = len(tokenizer(element)["input_ids"])
        current_length += element_token_length + delimiter_token_length
        
        if current_length <= max_length:
            filtered_elements.append(element)
        else:
            break
    return filtered_elements
