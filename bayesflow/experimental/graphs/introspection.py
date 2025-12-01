
# required methods

# method to distribute inference variables to individual networks
# method to identify the number of required inference networks
# method to identify the number of required summary networks
# method to identify if a node is able to be amortized
# method to identify which inference network needs which conditions
# method to identify if the conditions are group-wise or combined

# maps node names of inverted graph to node names in corresponding SimulationGraph
def original_node_names(inverted_graph):
    mapping = {}
    
    for node in inverted_graph.nodes:
        expanded_node = inverted_graph.expanded_graph.nodes[node]
        
        if expanded_node["merged_from"] != []:
            mapping[node] = expanded_node["merged_from"][0]
        elif expanded_node["previous_names"] == []:
            mapping[node] = node
        else:
            mapping[node] = expanded_node["previous_names"][0]

    return mapping
