import json
from pathlib import Path

from trying_graph_rag.graph_rag.types import GraphIndex, RelevantPointToQuery, SummarizedCommunity
from trying_graph_rag.utils import filter_non_fittable_elements, flatten, generate_ollama_response

PROMPT_DIR = Path(__file__).parent / "prompts"

with open(PROMPT_DIR / "map.txt") as file:
    MAP_PROMPT = file.read()

with open(PROMPT_DIR / "reduction.txt") as file:
    REDUCTION_PROMPT = file.read()


DEFAULT_MAP_PROMPT_LENGTH = len(MAP_PROMPT.format(query="", community_report=""))
DEFAULT_REDUCTION_PROMPT_LENGTH = len(REDUCTION_PROMPT.format(query="", relevant_points=""))

def map_query_to_community(
    query: str, community_report: SummarizedCommunity
) -> tuple[SummarizedCommunity, list[RelevantPointToQuery]]:
    # for each community report, generate a partial answer using the query as context
    # calculate the relevance score of the partial answer
    # return a list of community reports with their relevance scores
    def parse_output(ollama_response: str) -> list[RelevantPointToQuery]:
        ollama_response = ollama_response.strip()
        if ollama_response.startswith("```json") and ollama_response.endswith("```"):
            ollama_response = ollama_response[7:-3].strip()
        output_json = json.loads(ollama_response)
        return [RelevantPointToQuery(output_record) for output_record in output_json["points"]]

    # FIXME: this assumes that community report does not overflow the prompt length
    formatted_community_report = (
        community_report.community_report.model_dump_json()
    )  # just dump the community report to the prompt
    ollama_response = generate_ollama_response(
        prompt=MAP_PROMPT.format(query=query, community_report=formatted_community_report),
    )

    discovered_points = parse_output(ollama_response)

    return (community_report, discovered_points)


def reduce_to_one_answer(query: str, relevant_points: list[RelevantPointToQuery], top_n: int) -> str:
    """
    - sort the community_relevance_pairs by relevance score
    - combine the summaries of the top n communities together to form the final answer
    """
    non_zero_relevance_pairs = [relevant_point for relevant_point in relevant_points if relevant_point.score > 0]
    top_n = min(top_n, len(non_zero_relevance_pairs))
    
    sorted_relevant_points = sorted(non_zero_relevance_pairs, key=lambda x: x[1], reverse=True)
    top_n_relevant_points = sorted_relevant_points[:top_n]
    top_n_relevant_points_stringified = [relevant_point.model_dump_json() for relevant_point in top_n_relevant_points]
    fittable_relevant_points = filter_non_fittable_elements(top_n_relevant_points_stringified, max_length=7000-DEFAULT_REDUCTION_PROMPT_LENGTH, delimiter="\n")  # allocate about 1000 tokens for the output

    # TODO: maybe add community information as well (right now, we are only using the partial answers)
    concatenated_relevant_points = "\n".join(fittable_relevant_points)

    ollama_response = generate_ollama_response(
        prompt=REDUCTION_PROMPT.format(
            query=query,
            relevant_points=concatenated_relevant_points,
        ),
    )

    return ollama_response


def query_index(query: str, index: GraphIndex, hierarchy_level: int = 0, top_n: int = 5) -> str:
    # load the index, for the particular hierachy level
    if hierarchy_level not in index.hierachical_communities:
        raise ValueError(f"Invalid hierachy level: {hierarchy_level}")

    corresponding_level_communities = index.hierachical_communities[hierarchy_level]
    community_wise_relevant_points = [
        map_query_to_community(query, community) for community in corresponding_level_communities
    ]
    # just get the relevant points for now (ignore SummarizedCommunity)
    all_relevant_points = flatten([relavent_points for _, relavent_points in community_wise_relevant_points])
    return reduce_to_one_answer(query, all_relevant_points, top_n=top_n)
