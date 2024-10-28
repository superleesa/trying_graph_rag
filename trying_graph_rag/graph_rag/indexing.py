import networkx as nx

from graspologic.partition import hierarchical_leiden


def extract_entities_and_relations(text: str) -> tuple[list[dict], list[dict]]:
    # use the generate_relationships.txt as a prompt
    # and use the llm to generate the entities and relationships
    pass

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
    
    
    
    