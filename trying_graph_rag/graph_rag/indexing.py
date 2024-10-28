import networkx as nx

import ollama
from graspologic.partition import hierarchical_leiden

from trying_graph_rag.graph_rag.types import Entity, Relationship


with open("../trying_graph_rag/graph_rag/prompts/entities_and_relationships_extraction.txt") as file:
    PROMPT = file.read()


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
                    entity_name=record_content[1],
                    entity_type=record_content[2],
                    entity_description=record_content[3]
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
                    relationship_description=record_content[3],
                    relationship_strength=relationship_strength
                )
                relationships.append(relationship)
            else:
                print(f"Invalid record format: {record}")
        
        return entities, relationships
    
    ollama_response = ollama.generate(model='gemma2:2b', prompt=PROMPT.format(input_text=document, entity_types=str(entity_types)[1:-1], tuple_delimiter=tuple_delimiter, record_delimiter=record_delimiter, completion_delimiter=completion_delimiter), options={"temperature": 0})
    
    content = ollama_response['response']
    return parse_output(content, tuple_delimiter, record_delimiter, completion_delimiter)
 

def summarize_entities(entities: list[dict]) -> str:
    # basically you just put description of all entities into llm
    # TODO: look for a prompt that summarizes entities in the graph rag repo
    pass


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


def merge_entities(entities: list[dict]) -> dict:
    pass


def create_index(documents: list[str]) -> None:
    
    entities, relationships = zip([extract_entities_and_relations(doc) for doc in documents])  # i think entities should keep track of original document
    # for now ignore relationships
    entities = merge_entities(entities)
    entity_summaries = summarize_entities(entities)  # maybe the entities themselves should have a summary
    
    # create a graph of entities
    graph = nx.Graph()
    graph.add_nodes_from(entities)
    graph.add_edges_from(relationships)
    
    # TODO: create communities but with different hierarchies
    possible_hierarchy_levels = 3
    communities_in_hierachy = [create_communities(graph, threshold) for threshold in range(possible_hierarchy_levels)]
    
    # store the index
    index = {
        "entities": entities,
        "entity_summaries": entity_summaries,
        "communities": communities_in_hierachy
    }
    
    
    
    