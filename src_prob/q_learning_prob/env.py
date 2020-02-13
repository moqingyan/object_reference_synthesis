import torch 
import os 
import sys 
import json 

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from embedding import Data
from scene2graph import Graph
from cmd_args import cmd_args, logging
from query import SceneInterp
from utils import get_reward, get_final_reward, Encoder, NodeType, EdgeType, get_all_clauses, get_config, get_reachable_dict
from copy import deepcopy

# A mini version of env, storing information needed for decoding 
class State():

    def __init__(self, actions, data, graph, state, action_dict, idx_selected):
        self.data = data
        self.graph = graph
        self.actions = actions
        self.idx_selected = idx_selected
        self.action_dict = action_dict
        
    def get_clauses_idx(self): 

        names = self.graph.get_nodes()
        name_dict = {}
        for name_id, name in enumerate(names):
            name_dict[name] = name_id 
        
        unary_clauses_idx = []
        binary_clauses_idx = []

        for clause in self.actions:
            clause_idx = []

            if not clause[0] == "right" or clause[0] == "left":
                for element in clause[1:]:
                    if type(element) == int:
                        element = f"var_{element}"
                    clause_idx.append(name_dict[element])

                unary_clauses_idx.append(clause_idx)

            else:
                for element in clause:
                    if type(element) == int:
                        element = f"var_{element}"

                    if element == "right":
                        element = "center_right"
                    if element == "behind":
                        element = "center_behind"
                        
                    clause_idx.append(name_dict[element])
                binary_clauses_idx.append(clause_idx)

        return unary_clauses_idx, binary_clauses_idx

class Env():

    def __init__(self, data, graph, config, attr_encoder, state=None, ref=False, is_uncertain=False):
        
        self.data = deepcopy(data)
        self.graph = deepcopy(graph)
        self.obj_nums = len(graph.scene["objects"])
        self.clauses = []
        self.idx_selected = []

        self.config = config
        self.attr_encoder = attr_encoder
        self.actions = get_all_clauses(config)
        self.create_action_dict()
        self.ref_flag = ref 
        if ref:
            self.ref = [0]
        else: 
            self.ref = list(range(cmd_args.max_var_num))
        
        # self.obj_poss_left = [ self.obj_nums ** cmd_args.max_var_num ]
        self.obj_poss_left = [self.obj_nums]
        self.success = False
        self.possible = True
        self.unreachable = []
        self.reachable_dict, self.unreachable_dict = get_reachable_dict(self.actions)
        self.is_uncertain = is_uncertain
        if is_uncertain:
            self.interp = SceneInterp(graph.scene, config, is_uncertain=True)
        else:
            self.interp = SceneInterp(graph.scene["ground_truth"], config, is_uncertain=False)

        if type(state) == type(None):
            self.state = self.interp.get_init_state()
        else:
            self.state = state

    # def reset(self, graph):
    #     self.update_data(graph, self.attr_encoder)

    def create_action_dict(self):
        self.action_dict = {}
        for action_id, action in enumerate(self.actions):
            self.action_dict[str(action)] = action_id 

    # TODO: check the effect of update data on the whole dataset
    def update_data(self, binding_dict):
        self.graph.update_binding(binding_dict)
        # self.data.update_data(self.graph, self.attr_encoder)

        x = self.attr_encoder.get_embedding( self.graph.get_nodes())
        edge_index, edge_types = self.graph.get_edge_info()
        # edge_attrs = torch.tensor(self.attr_encoder.get_embedding(edge_types))
        edge_attr = torch.tensor(edge_types)
        edge_index = torch.tensor(edge_index)
        x = torch.tensor(x)
        batch = torch.zeros(x.shape[0], dtype=torch.int64)

        # print(f"previous edge num: {len(self.data.edge_attr)}")
        self.data = Data(x=x, y=self.data.y, edge_index=edge_index, edge_attr=edge_attr)
        self.data.batch = batch
        # print(f"previous edge num: {len(self.data.edge_attr)}")
        
    def check_success(self, binding_dict):
        if not self.obj_poss_left[-1] == 1:
            return 

        if "var_0" not in binding_dict.keys():
            return

        if list(binding_dict["var_0"])[0] == int(self.data.y):
            # print("success!")
            self.success = True
    
    def check_possible(self, binding_dict):
        if self.success:
            return 
        
        if "var_0" not in binding_dict.keys():
            return 
            
        if "var_0" in binding_dict.keys() and int(self.data.y) in binding_dict["var_0"]:
            return 

        self.possible = False 
    
    def get_state(self, unreachable = None):
        

        self.unreachable = self.unreachable_dict[str(sorted(self.ref))]
        unreachable = self.idx_selected + self.unreachable
        return State(self.actions, self.data, self.graph, self.state, self.action_dict, unreachable)

    # not support batching yet
    def step(self, action_idx):
        
        is_uncertain=self.is_uncertain
        next_clause = self.actions[action_idx]

        if self.ref_flag:
            for element in next_clause:
                if type (element) == int:
                    if element not in self.ref:
                        self.ref.append(element)

            self.unreachable = self.unreachable_dict[str(sorted(self.ref))]

        # if 'red' in next_clause or 'blue' in next_clause:
        #     print('here')

        self.idx_selected.append(action_idx)
        self.clauses.append(next_clause)

        useful = self.interp.useful_check(self.state, next_clause)
        logging.info(f"useful: {useful}")
        binding_dict, new_state = self.interp.state_query(self.state, next_clause)
        self.state = new_state

        # update the success and possible fields
        self.check_possible(binding_dict)

        if not self.possible:
            self.obj_poss_left.append(0)
        elif not "var_0" in binding_dict.keys():
            self.obj_poss_left.append(self.obj_nums)
            self.update_data(binding_dict)
        else:
            self.obj_poss_left.append(len(binding_dict["var_0"]))
            self.update_data(binding_dict)

        # update whether done or not
        self.check_success(binding_dict)
        done = self.success or (not self.possible)

        if not useful:
            reward = torch.tensor(-1, dtype=torch.float32)
        else:
            reward = torch.tensor(get_reward(self), dtype=torch.float32)
        
        self.update_data(binding_dict)
        info = {}

        logging.info(f"selected: {self.idx_selected}")
        logging.info(f"done: {done}")
        logging.info(f"success: {self.success}")

        if self.ref_flag:
            state = self.get_state(self.unreachable)
        else:
            state = self.get_state()

        return state, reward, done, info

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
    data_dir = os.path.abspath(__file__ + "../../../../data")
    root = os.path.abspath(os.path.join(data_dir, "./processed_dataset"))
    

    config = get_config()
    
    attr_encoder = Encoder(config)

    scenes_path = os.path.abspath(os.path.join(data_dir, f"./processed_dataset/raw/{cmd_args.scene_file_name}"))
    with open (scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)

    # construct a mini example
    target_id = 0
    graph = Graph(config, scenes[0], target_id)
    
    x = attr_encoder.get_embedding(graph.get_nodes())
    edge_index, edge_types = graph.get_edge_info()
    edge_attrs = attr_encoder.get_embedding([f"edge_{tp}" for tp in edge_types])
    data_point = Data(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attrs), graph.target_id)

    # construct an env
    env = Env(data_point, graph, config, attr_encoder)
    # env.reset(graph)
    