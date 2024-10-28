import random

import ollama
from graspologic.partition import hierarchical_leiden
import networkx as nx
from transformers import AutoTokenizer

from trying_graph_rag.graph_rag.types import Entity, Relationship, UniqueEntity


with open("../trying_graph_rag/graph_rag/prompts/entities_and_relationships_extraction.txt") as file:
    ENTITIES_AND_RELATIONSHIPS_EXTRACTION_PROMPT = file.read()

with open("../trying_graph_rag/graph_rag/prompts/summarize_entities.txt") as file:
    SUMMARIZE_ENTITIES_PROMPT = file.read()
    
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
SUMMARIZA_ENTITIES_PROMPT_LENGTH = len(tokenizer(SUMMARIZE_ENTITIES_PROMPT.format(entity_name="", description_list=""))["input_ids"])

def extract_entities_and_relations(document: str, entity_types: list[str], tuple_delimiter: str = r'<>', record_delimiter: str = '\n', completion_delimiter: str = '###END###') -> tuple[list[Entity], list[Relationship]]:
    """
    use the generate_relationships.txt as a prompt
    and use the llm to generate the entities and relationships
    """
    
    def parse_output(output: str, tuple_delimiter: str = r'<>', record_delimiter: str = '\n', completion_delimiter: str = '###END###') -> tuple[list[Entity], list[Relationship]]:
        output = output.replace(completion_delimiter, '')
        records = output.strip().split(record_delimiter)
        
        entities: list[Entity] = []
        relationships: list[Relationship] = []
        
        for record in records:
            record = record.strip().lstrip('(').rstrip(')')
            # skip empty records
            if not record:
                continue
            
            record_content = [record_field.strip().strip("'\"") for record_field in record.split(tuple_delimiter)]
            
            # skip empty records pt2
            if not len(record_content) or not all(record_content):
                continue
            
            record_type = record_content[0]
            if record_type not in ['entity', 'relationship']:
                raise ValueError(f"Invalid record type: {record_content}")
            
            if record_type == 'entity' and len(record_content) == 4:
                entity = Entity(
                    name=record_content[1],
                    type=record_content[2],
                    description=record_content[3]
                )
                entities.append(entity)
            elif record_type == 'relationship' and len(record_content) == 5:
                try:
                    relationship_strength = int(record_content[4])
                except ValueError:
                    raise ValueError("Invalid relationship strength")
                
                relationship = Relationship(
                    source_entity=record_content[1],
                    target_entity=record_content[2],
                    description=record_content[3],
                    strength=relationship_strength
                )
                relationships.append(relationship)
            else:
                print(f"Invalid record format: {record}")
        
        return entities, relationships
    
    ollama_response = ollama.generate(model='gemma2:2b', prompt=ENTITIES_AND_RELATIONSHIPS_EXTRACTION_PROMPT.format(input_text=document, entity_types=str(entity_types)[1:-1], tuple_delimiter=tuple_delimiter, record_delimiter=record_delimiter, completion_delimiter=completion_delimiter), options={"temperature": 0})
    
    content = ollama_response['response']
    return parse_output(content, tuple_delimiter, record_delimiter, completion_delimiter)
 

def summarize_grouped_entities(entities: list[Entity]) -> UniqueEntity:
    entity_name = entities[0].name
    entity_type = entities[0].type
    
    shuffled_entities = random.sample(entities, len(entities))  # shuffle to avoid bias
    descriptions = [entity.description for entity in shuffled_entities]
    
    # need to enusure that all the entity decription fit within the llm context length
    # gemma2:2b has a context length of 8k (8192 tokens)
    total_tokens = SUMMARIZA_ENTITIES_PROMPT_LENGTH
    selected_descriptions = []
    for i, description in enumerate(descriptions):
        total_tokens += len(tokenizer(description)["input_ids"]) + (2 if i != len(descriptions)-1 else 0)  # +2 for the comma and space
        if total_tokens > 8192:
            break
        
        selected_descriptions.append(description)
    
    concatenated_descriptions = ', '.join(selected_descriptions)
    ollama_response = ollama.generate(model='gemma2:2b', prompt=SUMMARIZE_ENTITIES_PROMPT.format(entity_name=entity_name, description_list=concatenated_descriptions), options={"temperature": 0})['response']
    return UniqueEntity(name=entity_name, type=entity_type, summary=ollama_response["response"])


def summarize_communities(communities: list[dict]) -> str:
    # i'm not sure how the communities would look like
    # it's probably a graph


def create_communities(graph: nx.Graph, threshold: float) -> dict[int, dict[str, int]]:
    # use leiden algorithm to create communities
    # adpoted from: https://github.com/microsoft/graphrag/blob/083de12bcf9e68e51032500106dc4aff771445a4/graphrag/index/operations/cluster_graph.py#L183

    community_mapping = hierarchical_leiden(
        graph, max_cluster_size=max_cluster_size, random_seed=seed
    )
    results: dict[int, dict[str, int]] = {}
    for partition in community_mapping:
        results[partition.level] = results.get(partition.level, {})
        results[partition.level][partition.node] = partition.cluster

    return results


def merge_same_name_entities(entities: list[Entity]) -> dict[str, list[Entity]]:
    enity_name_to_entities: dict[str, list[Entity]] = {}
    
    for entity in entities:
        key = entity.name+entity.type
        enity_name_to_entities[key] = enity_name_to_entities.get(key, [])
        enity_name_to_entities[key].append(entity)


def create_graph(entities: list[UniqueEntity], relationships: list[Relationship]) -> nx.Graph:
    #　FIXME: this ignores the type of the entity
    
    # use index within the entities list as the node id
    entity_to_id = {entity.name: i for i, entity in enumerate(entities)}
    graph_temp = [[0 for _ in range(len(entities))] for _ in range(len(entities))]
    for relationship in relationships:
        source_id = entity_to_id[relationship.source_entity]
        target_id = entity_to_id[relationship.target_entity]
        strength = relationship.strength
        graph_temp[source_id][target_id] += strength
        graph_temp[target_id][source_id] += strength
    
    graph = nx.Graph()
    graph.add_nodes_from([entity.name for entity in entities])
    for source_id in range(len(entities)):
        for target_id in range(source_id, len(entities)):
            if graph_temp[source_id][target_id] > 0:
                graph.add_edge(entities[source_id].name, entities[target_id].name, strength=graph_temp[source_id][target_id])
    
    return graph


def create_index(documents: list[str]) -> None:
    
    entities, relationships = zip([extract_entities_and_relations(doc) for doc in documents])  # i think entities should keep track of original document
    # for now ignore relationships
    grouped_entities = merge_same_name_entities(entities)
    unique_entities = [summarize_grouped_entities(entities) for _, entities in grouped_entities]  # maybe the entities themselves should have a summary
    graph = create_graph(unique_entities, relationships)
    
    # TODO: create communities but with different hierarchies
    possible_hierarchy_levels = 3
    communities_in_hierachy = [create_communities(graph, threshold) for threshold in range(possible_hierarchy_levels)]
    
    # store the index
    index = {
        "entities": entities,
        "entity_summaries": unique_entities,
        "communities": communities_in_hierachy
    }
    
    
    
    