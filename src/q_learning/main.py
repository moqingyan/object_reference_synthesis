import os 
import sys
import json
import pickle

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from embedding import GNN, SceneDataset, GNNLocal, GNNGL, GNNGlobal
from scene2graph import Graph, GraphNode
from cmd_args import cmd_args, logging
from utils import NodeType, EdgeType, Encoder, get_config, get_all_clauses, get_final_reward
from training import train

if __name__ == "__main__":

    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, cmd_args.scene_file_name))
    graphs_path = os.path.join(raw_path, cmd_args.graph_file_name)

    with open(scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)
    print("HEREEEEE")
    print(cmd_args.scene_file_name)
    print(len(scenes))

    config = get_config()

    graphs = []
    attr_encoder = Encoder(config)

    for scene in scenes:
        for target_id in range(len(scene["objects"])):
            graph = Graph(config, scene, target_id)
            graphs.append(graph)
    
    with open(graphs_path, 'wb') as graphs_file:
        pickle.dump(graphs, graphs_file) 

    root = os.path.join(data_dir, "./processed_dataset")
    scene_dataset = SceneDataset(root, config)

    train(scene_dataset, graphs, config)