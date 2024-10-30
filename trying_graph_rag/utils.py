import re
from typing import TypeVar

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


T = TypeVar("T")
def flatten(tuple_list: tuple[list[T]]) -> list[T]:
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


class SafeFormatter:
    """
    Safe string formatter that does not raise KeyError if key is missing.
    Ported from llama index
    at: https://github.com/run-llama/llama_index/blob/main/llama-index-core/llama_index/core/prompts/utils.py
    """

    def __init__(self, format_dict: dict[str, str] | None = None):
        self.format_dict = format_dict or {}

    def format(self, format_string: str) -> str:
        return re.sub(r"\{([^{}]+)\}", self._replace_match, format_string)

    def parse(self, format_string: str) -> list[str]:
        return re.findall(r"\{([^{}]+)\}", format_string)

    def _replace_match(self, match: re.Match) -> str:
        key = match.group(1)
        return str(self.format_dict.get(key, match.group(0)))


def safe_format_prompt(prompt: str, **kwargs: str) -> str:
    """Format a string with kwargs."""
    formatter = SafeFormatter(format_dict=kwargs)
    return formatter.format(prompt)
