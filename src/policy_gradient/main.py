import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time 

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from cmd_args import cmd_args, logging
from scene2graph import Graph, GraphNode
from embedding import GNN, SceneDataset
from torch_geometric.data import Data, DataLoader
from torch.autograd import Variable
from torch import autograd
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from query import query
from utils import AVAILABLE_OBJ_DICT, policy_gradient_loss, get_reward, get_final_reward, get_config
from env import Env 
from decoder import NodeDecoder
from rl import fit, test, RefRL
from utils import NodeType, EdgeType, Encoder

if __name__ == "__main__":
    
    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, cmd_args.scene_file_name))
    graphs_path = os.path.join(raw_path, cmd_args.graph_file_name)

    # In the pytorch geometry package, only int and tensor seems to be allowed to save
    # we process all the graphs and save them to a file.
    
    with open(scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)

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

    if os.path.exists(cmd_args.model_path) and os.path.getsize(cmd_args.model_path) > 0:
        refrl = torch.load(cmd_args.model_path)
        logging.info("Loaded refrl model")
    else:
        refrl = RefRL(scene_dataset, config, graphs)
        logging.info("Constructed refrl model")

    start_time = time.time()
    logging.info(f"Start {cmd_args.phase}")
    if cmd_args.phase == "training": 
        fit(refrl)
    else:
        test (refrl, "train")
        test (refrl, "test")
    end_time = time.time()
    logging.info(f"Finished {cmd_args.phase} in {end_time - start_time}")

    print("Done")
    


    
    
    