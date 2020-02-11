# For graph visualization
# libraries
import pickle as pkl
import os

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
 
from scene2graph import GraphNode, Edge, Graph

def process_node(graph):
    node_info = dict()
    node_info["ID"] = []
    node_info["node_type"] = []
    for node in graph.nodes: 
        if node.name not in node_info["ID"]:
            node_info["ID"].append(node.name) 
            node_info["node_type"].append(node.node_type.name)
    return node_info

def process_edge(graph):
    edge_info = dict()
    edge_info["from"] = []
    edge_info["to"] = []
    edge_info["value"] = []
    for edge in graph.edges:
        edge_info["from"].append(graph.nodes[edge.src_idx].name)
        edge_info["to"].append(graph.nodes[edge.dst_idx].name)
        edge_info["value"].append(edge.edge_type.name) 
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
    nx.draw(G, with_labels=True, node_color=carac['node_type'].cat.codes, cmap=plt.cm.Set1, node_size=1500)
    plt.show()

if __name__ == "__main__":
    data_dir = os.path.abspath(__file__ + "../../../../data")
    graphs_path = os.path.abspath( os.path.join(data_dir, "./processed_dataset/raw/unit_test_3_graphs.pkl"))

    with open(graphs_path, 'rb') as graphs_file:
        graphs = pkl.load(graphs_file)
    
    draw_graph(graphs[0])