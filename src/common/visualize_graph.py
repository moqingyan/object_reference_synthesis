# libraries
import pickle as pkl
import os
from enum import Enum, unique

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
 
from scene2graph import GraphNode, Edge, Graph

class NodeType(Enum):
    obj = 10
    var = 1
    attr = 2
    relation = 3 
    target = 4 
    center_relation = 5
    first = 6 
    second = 7
    gb = 8
    binding = 9

def get_node_type(name):
    if "obj" in name:
        return NodeType.obj 
    if "var" in name:
        return NodeType.var 
    if "left" in name or "right" in name or "behind" in name or "front" in name:
        if "center" in name:
            return NodeType.center_relation
        else:
            return NodeType.relation
    if "target" in name:
        return NodeType.target
    return NodeType.attr
    
def process_node(graph):
    node_info = dict()
    node_info["ID"] = []
    node_info["node_type"] = []
    nodes = graph.get_nodes()
    for node in nodes: 

        if node not in node_info["ID"]:
            
            node_info["ID"].append(node)
            node_info["node_type"].append(get_node_type(node))

    return node_info

def subst(node):
    if node == "center_right":
        node = "right"
    if node == "center_behind":
        node = "behind"
    return node 

def process_edge(graph):
    edge_info = dict()
    edge_res = graph.get_edge_info()
    nodes = [subst(node) for node in graph.get_nodes()]

    edge_info["from"] = [nodes[f] for f in edge_res[0][0]]
    edge_info["to"] = [nodes[f] for f in edge_res[0][1]]
    edge_info["value"] = edge_res[1]

    return edge_info

def draw_graph(graph):
    node_info = process_node(graph)
    edge_info = process_edge(graph)
    print(f"node: {node_info}")
    print(f"edge: {edge_info}")

    # Build a dataframe with your connections
    df = pd.DataFrame(edge_info)
        
    # And a data frame with characteristics for your nodes
    carac = pd.DataFrame(node_info)
 
    # Build your graph
    G=nx.from_pandas_edgelist(df, 'from', 'to', create_using=nx.DiGraph() )
    
    # The order of the node for networkX is the following order:
    G.nodes()
    # Thus, we cannot give directly the 'myvalue' column to netowrkX, we need to arrange the order!
 
    # Here is the tricky part: I need to reorder carac to assign the good color to each node
    carac= carac.set_index('ID')
    carac=carac.reindex(G.nodes())
 
    # And I need to transform my categorical column in a numerical value: group1->1, group2->2...
    carac['node_type']=pd.Categorical(carac['node_type'])
    carac['node_type'].cat.codes
 
    # Custom the nodes:
    nx.draw(G, with_labels=True, node_color=carac['node_type'].cat.codes, cmap=plt.cm.Pastel2, node_size=1500)
    # plt.show()
    plt.savefig("graph.svg", format='svg')

if __name__ == "__main__":
    data_dir = os.path.abspath(__file__ + "../../../../data")
    graphs_path = os.path.abspath( os.path.join(data_dir, "./processed_dataset/raw/unit_test_3_graphs.pkl"))

    with open(graphs_path, 'rb') as graphs_file:
        graphs = pkl.load(graphs_file)
    
    draw_graph(graphs[0])