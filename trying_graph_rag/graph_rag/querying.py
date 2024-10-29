from pathlib import Path

import ollama

from trying_graph_rag.graph_rag.types import GraphIndex, SummarizedCommunity

PROMPT_DIR = Path(__file__).parent / "prompts"

with open(PROMPT_DIR / "map.txt") as file:
    MAP_PROMPT = file.read()
    
with open(PROMPT_DIR / "reduction.txt") as file:
    REDUCTION_PROMPT = file.read()


def map_query_to_community(query: str, community_reports: list[SummarizedCommunity]) -> list[tuple[SummarizedCommunity, str, float]]:
    # for each community report, generate a partial answer using the query as context
    # calculate the relevance score of the partial answer
    # return a list of community reports with their relevance scores
    def parse_output(output: str) -> tuple[str, float]:
        #TODO
        pass
    
    community_partial_answers = []
    
    for community_report in community_reports:
        ollama_response = ollama.generate(
            model="gemma2:2b",
            prompt=query + community_report.community_report.summary,
            options={"temperature": 0},
        )["response"]
        
        partial_answer, relevance_score = parse_output(ollama_response)
        
        community_partial_answers.append((community_report, partial_answer, relevance_score))
    
    return community_partial_answers


def reduce_answer(query: str, community_partial_answers: list[tuple[SummarizedCommunity, str, float]], top_n: int) -> str:
    # sort the community_relevance_pairs by relevance score
    # combine the summaries of the top 3 communities together to form the final answer
    non_zero_relevance_pairs = [(partial_answer, relevance_score) for _, partial_answer, relevance_score in community_partial_answers if relevance_score > 0]
    sorted_pairs = sorted(non_zero_relevance_pairs, key=lambda x: x[1], reverse=True)
    top_n_pairs = sorted_pairs[:top_n]  # TODO: make this customizable?
    
    # TODO: maybe add community information as well (right now, we are only using the partial answers)
    concatenated_community_reports = "\n".join([partial_answer for partial_answer, _ in top_n_pairs])
    
    ollama_response = ollama.generate(
        model="gemma2:2b",
        prompt=REDUCTION_PROMPT.format(
            query=query,
            community_reports=concatenated_community_reports,
        ),
        options={"temperature": 0},
    )["response"]
    
    return ollama_response

    

def query_index(query: str, index: GraphIndex, hierachy_level: int = 1, top_n: int = 3) -> str:
    # load the index, for the particular hierachy level
    corresponding_level_communities = index.hierachical_communities[hierachy_level]
    community_partial_answers = map_query_to_community(query, corresponding_level_communities)
    return reduce_answer(query, community_partial_answers, top_n=top_n)
