
# required methods

# method to distribute inference variables to individual networks
# method to identify the number of required inference networks
# method to identify the number of required summary networks
# method to identify if a node is able to be amortized
# method to identify which inference network needs which conditions
# method to identify if the conditions are group-wise or combined

def node_names_in_simulation_graph(inverted_graph):
    mapping = {}
    
    for node in inverted_graph.nodes:
        expanded_node = self.expanded_graph.nodes[node]
