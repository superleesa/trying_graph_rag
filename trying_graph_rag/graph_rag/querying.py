

def query_index(query: str, hierachy_level: int = 1) -> list[dict]:
    # load the index, for the particular hierachy level
    
    # for each community
    # add community summary as context together with the query -> let the model generate the partial answer + relavancy score
    
    # if the relavancy score is high enough, add the community to the list of possible answers
    # combine the partial answers together with the query to generate the final answer