# trying_graph_rag

- CPU-only GraphRag + Default RAG for Multi-Hop QA
- Uses qunatized (Q4) Gemma2 2B through Ollama


## Setup
```shell
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull gemma2:2b
huggingface-cli login  # for gemma 2 tokenizer access
```

## Indexing
```shell
poetry run python scripts/batch_index.py dataset/sampled_gold_data.json 
```

## Querying
```shell
poetry run python scripts/batch_query.py dataset/sampled_gold_data.json . dataset/graph_rag_prediction.csv
```