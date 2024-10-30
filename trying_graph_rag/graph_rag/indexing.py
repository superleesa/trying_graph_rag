import logging
import pickle
import random
from pathlib import Path

try:
    import pyjson5 as json  # use json5 for less strict json parsing (e.g. allows single quotes, trialing commas)
except Exception:
    # setuptools install `pyjson5` as `json5` (but poetry installs it as `pyjson5`)
    import json5 as json

import networkx as nx
from graspologic.partition import hierarchical_leiden
from tqdm import tqdm
from transformers import AutoTokenizer

from trying_graph_rag.graph_rag.types import (
    CommunityReport,
    Entity,
    GraphIndex,
    Relationship,
    SummarizedCommunity,
    SummarizedUniqueEntity,
)
from trying_graph_rag.utils import flatten, generate_ollama_response, safe_format_prompt

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("generated_contents.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)

NUM_MAX_TRIALS = 3

PROMPT_DIR = Path(__file__).parent / "prompts"

with open(PROMPT_DIR / "entities_and_relationships_extraction.txt") as file:
    ENTITIES_AND_RELATIONSHIPS_EXTRACTION_PROMPT = file.read()

with open(PROMPT_DIR / "entity_summarization.txt") as file:
    SUMMARIZE_ENTITIES_PROMPT = file.read()

with open(PROMPT_DIR / "community_report.txt") as file:
    COMMUNITY_REPORTS_PROMPT = file.read()

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
SUMMARIZA_ENTITIES_PROMPT_LENGTH = len(
    tokenizer(SUMMARIZE_ENTITIES_PROMPT.format(entity_name="", description_list=""))["input_ids"]
)


def extract_entities_and_relations(
    document: str,
    entity_types: list[str],
    tuple_delimiter: str = r"|",
    record_delimiter: str = "\n",
    completion_delimiter: str = "###END###",
) -> tuple[list[Entity], list[Relationship]]:
    """
    use the generate_relationships.txt as a prompt
    and use the llm to generate the entities and relationships
    """
    def parse_json_output(output: str) -> tuple[list[Entity], list[Relationship]]:
        output = output.strip()
        if output.startswith("```json") and output.endswith("```"):
            output = output[7:-3].strip()
        output_json = json.loads(output)
        
        unique_entities: dict[str, Entity] = {}
        relationships: list[Relationship] = []
        
        for output_record in output_json["entities"]:
            entity = Entity(**output_record)
            if entity.name in unique_entities:
                # skip duplicate entities
                continue
            unique_entities[entity.name] = entity
        
        for output_record in output_json["relationships"]:
            relationship = Relationship(**output_record)
            
            # if source / target entity is missing, add it as an entity, and use the relationship description as the entity description
            # this is not ideal but it's better than ignoring the relationship
            if relationship.source_entity not in unique_entities:
                entity = Entity(name=relationship.source_entity, type="UNKNOWN", description=relationship.description)
                unique_entities[relationship.source_entity] = entity
            if relationship.target_entity not in unique_entities:
                entity = Entity(name=relationship.target_entity, type="UNKNOWN", description=relationship.description)
                unique_entities[relationship.target_entity] = entity
            
            relationships.append(relationship)
        
        return list(unique_entities.values()), relationships

    ollama_response = generate_ollama_response(
        prompt=safe_format_prompt(ENTITIES_AND_RELATIONSHIPS_EXTRACTION_PROMPT, input_text=document, entity_types=entity_types),
    )
    logger.info(f"Generated entities and relationships: {ollama_response}")
    return parse_json_output(ollama_response, tuple_delimiter, record_delimiter, completion_delimiter)


def summarize_grouped_entities(entities: list[Entity]) -> SummarizedUniqueEntity:
    entity_name = entities[0].name
    entity_type = entities[0].type

    shuffled_entities = random.sample(entities, len(entities))  # shuffle to avoid bias
    descriptions = [entity.description for entity in shuffled_entities]

    # need to enusure that all the entity decription fit within the llm context length
    # gemma2:2b has a context length of 8k (8192 tokens)
    total_tokens = SUMMARIZA_ENTITIES_PROMPT_LENGTH
    selected_descriptions = []
    for i, description in enumerate(descriptions):
        total_tokens += len(tokenizer(description)["input_ids"]) + (
            2 if i != len(descriptions) - 1 else 0
        )  # +2 for the comma and space
        if total_tokens > 6000:  # allocate remaining for the output (the summary)  TODO: check the actual limit
            break

        selected_descriptions.append(description)

    concatenated_descriptions = ", ".join(selected_descriptions)
    ollama_response = generate_ollama_response(
        prompt=SUMMARIZE_ENTITIES_PROMPT.format(entity_name=entity_name, description_list=concatenated_descriptions),
    )
    logger.info(f"Generated entity summary: {ollama_response}")
    return SummarizedUniqueEntity(name=entity_name, type=entity_type, summary=ollama_response)


def format_communities_and_summarize(
    hierarchy_level: int,
    node_id_to_community_id: dict[str, int],
    entity_id_to_entity: dict[str, SummarizedUniqueEntity],
) -> list[SummarizedCommunity]:
    # group by community id
    communities: dict[int, list[str]] = {}
    for node_id, community_id in node_id_to_community_id.items():
        communities[community_id] = communities.get(community_id, [])
        communities[community_id].append(node_id)

    formatted_communities = []
    for community_id, community in communities.items():
        community_entities = {entity_id: entity_id_to_entity[entity_id] for entity_id in community}

        # FIXME: don't add everything to the context (it will cause context length to exceed the limit)
        # TODO: for now we ignore relationship when creating community report, but we should include them
        concatenated_community_entities = ", ".join([entity.summary for entity in community_entities.values()])

        # we retry generating the community report if it fails
        # (usually because of json formatting issues)
        # hoing that we get better sample from the llm
        num_trials = 0
        while num_trials < NUM_MAX_TRIALS:
            try:
                ollama_response: str = generate_ollama_response(
                    prompt=COMMUNITY_REPORTS_PROMPT.format(community_entities=concatenated_community_entities),
                )
                logger.info(f"Generated community report: {ollama_response}")

                ollama_response = ollama_response.strip()
                if ollama_response.startswith("```json") and ollama_response.endswith("```"):
                    ollama_response = ollama_response[7:-3].strip()

                community_report = CommunityReport(**json.loads(ollama_response))
            except Exception as e:
                logger.error(f"Error generating community report (trial {num_trials}): {e}")
                num_trials += 1
            else:
                formatted_community = SummarizedCommunity(
                    community_id=community_id, hierachy_level=hierarchy_level, community_report=community_report
                )
                formatted_communities.append(formatted_community)
                break
        else:
            logger.error(f"Tried {num_trials+1} but Failed to generate community report for community: {community_id}")

    return formatted_communities


def create_communities(
    graph: nx.Graph, max_cluster_size: int | None = None, random_seed: int = 123456789
) -> dict[int, dict[str, int]]:
    # use leiden algorithm to create communities
    # adpoted from: https://github.com/microsoft/graphrag/blob/083de12bcf9e68e51032500106dc4aff771445a4/graphrag/index/operations/cluster_graph.py#L183

    # hirarchical leiden works by applying leiden again if the cluster size goes over the max_cluster_size
    # hence, if the max_cluster_size is smaller than the number of nodes
    # there won't be any hirarchies (there would onbly be one level)
    if max_cluster_size is None:
        max_cluster_size = len(graph.nodes) // 4
        logger.info(f"Setting max_cluster_size to {max_cluster_size}")

    community_mapping = hierarchical_leiden(graph, max_cluster_size=max_cluster_size, random_seed=random_seed)
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

    return results


def merge_same_name_entities(entities: list[Entity]) -> dict[str, list[Entity]]:
    enity_name_to_entities: dict[str, list[Entity]] = {}

    for entity in entities:
        key = entity.name  # should have `+ entity.type` but for now ignore it for simplicity
        enity_name_to_entities[key] = enity_name_to_entities.get(key, [])
        enity_name_to_entities[key].append(entity)

    return enity_name_to_entities


def create_graph(entities: list[SummarizedUniqueEntity], relationships: list[Relationship]) -> nx.Graph:
    # FIXME: this ignores the type of the entity

    # use index within the entities list as the node id
    entity_to_id = {entity.name: i for i, entity in enumerate(entities)}
    graph_temp = [[0 for _ in range(len(entities))] for _ in range(len(entities))]
    for relationship in relationships:
        try:
            source_id = entity_to_id[relationship.source_entity]
            target_id = entity_to_id[relationship.target_entity]
        except KeyError:
            # note: llm sometimes generate relationship that is not in the entities list
            logger.warning(f"Invalid relationship: {relationship}")
            continue
        strength = relationship.strength
        graph_temp[source_id][target_id] += strength
        graph_temp[target_id][source_id] += strength

    graph = nx.Graph()
    graph.add_nodes_from([entity.name for entity in entities])
    for source_id in range(len(entities)):
        for target_id in range(source_id, len(entities)):
            if graph_temp[source_id][target_id] > 0:
                graph.add_edge(
                    entities[source_id].name, entities[target_id].name, strength=graph_temp[source_id][target_id]
                )

    return graph


def create_index(documents: list[str], entity_types: list[str], index_name: str = "index") -> None:
    # TODO: don't use the same index id (maybe use a timestamp)

    _entities, _relationships = zip(
        *[extract_entities_and_relations(doc, entity_types) for doc in tqdm(documents, desc="Extracting entities")]
    )  # TODO: i think entities should keep track of original document??
    entities, relationships = flatten(_entities), flatten(_relationships)

    # TODO: for now we only summarize entities but we should also summarize relationships
    grouped_entities = merge_same_name_entities(entities)
    unique_id_to_entity = {
        entity_id: summarize_grouped_entities(entities)
        for entity_id, entities in tqdm(grouped_entities.items(), desc="Summarizing entities")
    }

    graph = create_graph(list(unique_id_to_entity.values()), relationships)
    hierarchical_communities = create_communities(graph, max_cluster_size=None, random_seed=123456789)

    summarized_communities = {
        hierarchical_level: format_communities_and_summarize(
            hierarchical_level, node_id_to_community_id, unique_id_to_entity
        )
        for hierarchical_level, node_id_to_community_id in tqdm(
            hierarchical_communities.items(), desc="Summarizing communities"
        )
    }

    index = GraphIndex(
        all_entities=entities,
        all_relationships=relationships,
        unique_entities=list(unique_id_to_entity.values()),
        hierachical_communities=summarized_communities,
    )

    with open(f"{index_name}.pickle", "wb") as file:
        pickle.dump(index, file)
