import re
from typing import TypeAlias
import copy
import networkx as nx

Node: TypeAlias = str


def split_node(graph: nx.DiGraph, node: Node) -> nx.DiGraph:
    subgraph = extract_subgraph(graph, node)
    other_nodes = set(graph.nodes).difference(subgraph.nodes)
    split_graph = nx.DiGraph(graph.subgraph(other_nodes))

    for i in [1, 2]:
        sorted_nodes = list(nx.topological_sort(subgraph))
        renamed_nodes = [add_suffix(n, suffix=i) for n in sorted_nodes]

        # add nodes from subgraph to new graph
        for original, renamed in zip(sorted_nodes, renamed_nodes):
            split_graph.add_node(renamed)

            # add metadata
            split_graph = add_split_metadata(split_graph, subgraph, original, renamed, node)

            split_graph = add_previous_names_metadata(split_graph, subgraph, original, renamed)

        # add edges
        for parent in graph.predecessors(node):
            split_graph.add_edge(parent, renamed_nodes[0])

        for start, stop in subgraph.edges:
            split_graph.add_edge(add_suffix(start, i), add_suffix(stop, i))

        # mend broken connections (ancestors of descendants of 'node')
        for start, stop in graph.edges:
            if start in other_nodes and stop in subgraph.nodes:
                split_graph.add_edge(start, add_suffix(stop, i))

    return nx.DiGraph(split_graph)


def extract_subgraph(graph: nx.DiGraph, node: Node) -> nx.DiGraph:
    included_nodes = set([node])
    included_nodes.update(nx.descendants(graph, node))

    subgraph = graph.subgraph(included_nodes)

    return nx.DiGraph(subgraph)


def has_open_path(graph: nx.DiGraph, x: Node, y: Node, known: list[Node]) -> bool:
    all_paths = list(nx.all_simple_paths(graph.to_undirected(), x, y))
    is_blocked = [False for _ in all_paths]

    for i, path in enumerate(all_paths):
        # If a node in the path is known and not a collider, it blocks that path.
        # If a node in the path is a collider and not known, it blocks that path.
        for idx, node in enumerate(path[1:-1]):
            is_collider = graph.has_edge(path[idx], path[idx + 1]) & graph.has_edge(path[idx + 2], path[idx + 1])
            if node in known and not is_collider:
                is_blocked[i] = True
            elif is_collider:
                known_descendant = len(nx.descendants(graph, node) & set(known)) != 0
                if node not in known and not known_descendant:
                    is_blocked[i] = True

    if all(is_blocked):
        return False
    else:
        return True


def add_suffix(string: str, suffix: int):
    if bool(re.search(r"_\d+$", string)):
        return string + str(suffix)
    else:
        return string + "_" + str(suffix)


def add_split_metadata(
    graph: nx.DiGraph,
    subgraph: nx.DiGraph,
    original: Node,
    renamed: Node,
    split_node: Node,
) -> nx.DiGraph:
    graph = graph.copy()

    if "split_by" in subgraph.nodes[original]:
        # transfer metadata on subgraph to graph
        splits = subgraph.nodes[original]["split_by"]
        graph.nodes[renamed]["split_by"] = splits

        # add split_node to split_on
        if split_node not in splits:
            graph.nodes[renamed]["split_by"].append(split_node)
    else:
        graph.nodes[renamed]["split_by"] = [split_node]

    return graph


def add_previous_names_metadata(graph: nx.DiGraph, subgraph: nx.DiGraph, original: Node, renamed: Node) -> nx.DiGraph:
    graph = graph.copy()

    if "previous_names" in subgraph.nodes[original]:
        # transfer metadata on subgraph to graph
        previous_names = subgraph.nodes[original]["previous_names"]
        graph.nodes[renamed]["previous_names"] = previous_names

        # add name to previous_names
        if original not in previous_names:
            graph.nodes[renamed]["previous_names"].append(original)
    else:
        graph.nodes[renamed]["previous_names"] = [original]

    return graph


def merge_root_nodes(graph: nx.DiGraph):
    root_nodes = [node for node in graph.nodes() if graph.in_degree(node) == 0]

    return merge_nodes(graph, root_nodes)


def merge_nodes(graph: nx.DiGraph, nodes: list[Node]):
    for node in nodes[1::]:
        graph = nx.contracted_nodes(graph, nodes[0], node, self_loops=False)

    new_name = ", ".join(map(str, nodes))
    graph = nx.relabel_nodes(graph, {nodes[0]: new_name}, copy=False)

    graph.nodes[new_name].clear()
    graph.nodes[new_name]["merged_from"] = nodes

    return graph
