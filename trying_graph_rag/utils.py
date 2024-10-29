import ollama
from ollama import Options
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
OLLAMA_GENERATION_CONFIG = Options(temperature=0, seed=123456789)

def generate_ollama_response(prompt: str) -> str:
    return ollama.generate(
        model="gemma2:2b",
        prompt=prompt,
        options=OLLAMA_GENERATION_CONFIG,
    )["response"]


def flatten[T](tuple_list: tuple[list[T]]) -> list[T]:
    return [item for sublist in tuple_list for item in sublist]