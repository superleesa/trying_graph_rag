# trying_graph_rag

- CPU-only GraphRag + Default RAG for Multi-Hop QA
- Uses qunatized (Q4) Gemma2 2B through Ollama

## Indexing
```shell
poetry run python scripts/batch_index.py dataset/sampled_gold_data.json 
```

## Querying
```shell
poetry run python scripts/batch_query.py dataset/sampled_gold_data.json . dataset/graph_rag_prediction.csv
```