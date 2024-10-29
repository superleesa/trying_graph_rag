import json
from pathlib import Path

from trying_graph_rag.graph_rag.types import GraphIndex, RelevantPointToQuery, SummarizedCommunity
from trying_graph_rag.utils import flatten, generate_ollama_response

PROMPT_DIR = Path(__file__).parent / "prompts"

with open(PROMPT_DIR / "map.txt") as file:
    MAP_PROMPT = file.read()

with open(PROMPT_DIR / "reduction.txt") as file:
    REDUCTION_PROMPT = file.read()


def map_query_to_community(
    query: str, community_reports: list[SummarizedCommunity]
) -> list[tuple[SummarizedCommunity, list[RelevantPointToQuery]]]:
    # for each community report, generate a partial answer using the query as context
    # calculate the relevance score of the partial answer
    # return a list of community reports with their relevance scores
    def parse_output(ollama_response: str) -> list[RelevantPointToQuery]:
        ollama_response = ollama_response.strip()
        if ollama_response.startswith("```json") and ollama_response.endswith("```"):
            ollama_response = ollama_response[7:-3].strip()
        output_json = json.loads(ollama_response)
        return [RelevantPointToQuery(output_record) for output_record in output_json["points"]]

    community_partial_answers = []

    for community_report in community_reports:
        formatted_community_report = (
            community_report.community_report.model_dump_json()
        )  # just dump the community report to the prompt
        ollama_response = generate_ollama_response(
            prompt=MAP_PROMPT.format(query=query, community_report=formatted_community_report),
        )

        discovered_points = parse_output(ollama_response)

        community_partial_answers.append((community_report, discovered_points))

    return community_partial_answers


def reduce_to_one_answer(query: str, relevant_points: list[RelevantPointToQuery], top_n: int) -> str:
    # sort the community_relevance_pairs by relevance score
    # combine the summaries of the top 3 communities together to form the final answer
    non_zero_relevance_pairs = [relevant_point for relevant_point in relevant_points if relevant_point.score > 0]
    sorted_pairs = sorted(non_zero_relevance_pairs, key=lambda x: x[1], reverse=True)
    top_n_pairs = sorted_pairs[:top_n]  # TODO: make this customizable?

    # TODO: maybe add community information as well (right now, we are only using the partial answers)
    concatenated_relevant_points = "\n".join([relevant_point.model_dump_json() for relevant_point in top_n_pairs])

    ollama_response = generate_ollama_response(
        prompt=REDUCTION_PROMPT.format(
            query=query,
            relevant_points=concatenated_relevant_points,
        ),
    )

    return ollama_response


def query_index(query: str, index: GraphIndex, hierachy_level: int = 1, top_n: int = 3) -> str:
    # load the index, for the particular hierachy level
    if hierachy_level not in index.hierachical_communities:
        raise ValueError(f"Invalid hierachy level: {hierachy_level}")

    corresponding_level_communities = index.hierachical_communities[hierachy_level]
    community_partial_answers = [
        map_query_to_community(query, community) for community in corresponding_level_communities
    ]
    # just get the relevant points for now (ignore SummarizedCommunity)
    all_relevant_points = flatten([relavent_points for _, relavent_points in community_partial_answers])
    return reduce_to_one_answer(query, all_relevant_points, top_n=top_n)
