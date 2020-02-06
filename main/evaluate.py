import json
import os
import torch
import sys
import logging
import pickle
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

common_path = os.path.abspath(os.path.join(__file__, "../../src/common"))
src_path = os.path.abspath(os.path.join(__file__, "../../src"))
sys.path.append(common_path)
sys.path.append(src_path)

from rl import fit, test, RefRL
from cmd_args import cmd_args
from utils import get_config
from scene2graph import Graph, GraphNode
from embedding import GNN, SceneDatase, create_dataset
from utils import AVAILABLE_OBJ_DICT, policy_gradient_loss, get_reward, get_final_reward,  NodeType, EdgeType, Encoder

def test_dataset(refrl, graphs, dataset):
    
    refrl.policy.eval()
    data_loader = DataLoader(dataset)
    
    success = {}
    for problem_id in range(len(data_loader)):
        success[problem_id] = 0

    eps = cmd_args.eps
    
    for it in range(cmd_args.test_iter):
        logging.info(f"testing iteration: {it}")
        with torch.no_grad():
            for data_point, ct in zip(data_loader, tqdm(range(len(data_loader)))):
                
                logging.info(ct)

                graph = graphs[data_point.graph_id]
                env, _ = refrl.episode(data_point, graph, eps, refrl.dataset.attr_encoder, phase="test")

                if env.success:
                    success[ct] += 1

    return success

if __name__ == "__main__":

    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    model_dir = os.path.abspath(os.path.join(data_dir, "oak_model/model"))
    log_dir = os.path.abspath(os.path.join(data_dir, "log"))

    scene_file_name = "img_test_500.json"
    graph_file_name = "img_test_500.pt"
    success_file_name = "success_500.json"

    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, scene_file_name))
    graphs_path = os.path.join(raw_path, graph_file_name)
    success_path = os.path.join(data_dir, f"./eval_result/{success_file_name}")

    # update the cmd_args corresponding to the info we have 
    cmd_args.graph_file_name = graph_file_name

    lr4_10_model_path = os.path.join(model_dir, "refrl-0.0001-penyes-no_intermediate-norno-img_test_30.pkl")
    # lr4_10_log_path = os.path.join(log_dir, "lr-0.0001-test_10.log")
    # print(lr4_10_log_path)
    # logging.basicConfig(filename=lr4_10_log_path, filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
    # logging.info("blah")
    
    refrl = torch.load(lr4_10_model_path)
    
    graphs, scene_dataset = create_dataset(scenes_path, graphs_path)
    # config = get_config()
    # refrl = RefRL(scene_dataset, config, graphs)

    success = test_dataset(refrl, graphs, scene_dataset)
    with open(success_path, 'w') as success_file:
        json.dump(success, success_file)

