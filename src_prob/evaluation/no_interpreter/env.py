import torch 
import os 
import sys 
import json 


common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)
from embedding import Data
from scene2graph import Graph, GraphNode, Edge, NodeType, EdgeType
from cmd_args import cmd_args
from query import query
from utils import get_reward, get_final_reward, Encoder, NodeType, EdgeType
from copy import deepcopy

class Env():

    def __init__(self, data, graph, config, attr_encoder):
        
        self.data = deepcopy(data)
        self.graph = deepcopy(graph)
        self.config = config
        self.attr_encoder = attr_encoder

        self.success = False
        self.possible = True 

        self.obj_poss_left = []
        self.state = None

    def check_success(self, binding_dict):
        if not self.obj_poss_left[-1] == 1:
            return 

        if "var_0" not in binding_dict.keys():
            return

        if binding_dict["var_0"][0] == str(int(self.data.y)):
            self.success = True
    
    def check_possible(self, binding_dict):
        if self.success:
            return 
        
        if "var_0" not in binding_dict.keys():
            return 
            
        if "var_0" in binding_dict.keys() and str(int(self.data.y)) in binding_dict["var_0"]:
            return 

        self.possible = False 

    def step(self):
        if len(self.clauses) == 0:
            return torch.tensor(0, dtype=torch.float32)
            
        binding_dict = query(self.graph.scene, self.clauses, self.config)

        # update the success and possible fields
        self.check_possible(binding_dict)

        if not self.possible:
            self.obj_poss_left.append(0)
        elif not "var_0" in binding_dict.keys():
            self.obj_poss_left.append(self.obj_nums)
            self.update_data(binding_dict)
        else:
            self.obj_poss_left.append( len(binding_dict["var_0"]))

        self.check_success(binding_dict)

        reward = get_reward(self)
        return torch.tensor(reward, dtype=torch.float32)

    def is_finished(self):
        # edge case handling
        if len(self.obj_poss_left) == 1:
            return False
    
        # # duplicate clauses
        # dupes = [x for n, x in enumerate(self.clauses) if x in self.clauses[:n]]
        # if len(dupes) > 0:
        #     return True 

        # no possibilities
        if self.obj_poss_left[-1] == 0 or self.obj_poss_left[-1] == 1 :
            return True

        # already succeed
        if self.success:
            return True

        # not possible
        if not self.possible:
            return True

        return False

if __name__ == "__main__":
    # load the data 
    data_dir = os.path.abspath(__file__ + "../../../data")
    root = os.path.abspath(os.path.join(data_dir, "./processed_dataset"))
    

    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    attr_encoder = Encoder(config)

    scenes_path = os.path.abspath(os.path.join(data_dir, f"./processed_dataset/raw/{cmd_args.scene_file_name}"))
    with open (scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)

    # construct a mini example
    target_id = 0
    graph = Graph(config, scenes[0], target_id)
    
    x = attr_encoder.get_embedding( [ node.name for node in graph.nodes])
    edge_index, edge_types = graph.get_edge_info()
    edge_attrs = torch.tensor(attr_encoder.get_embedding(edge_types))
    data_point = Data(x=x, edge_index=edge_index, edge_attr=edge_attrs, y=target_id)

    # construct an env
    env = Env(data_point, graph, config, attr_encoder)
    # env.reset(graph)
    