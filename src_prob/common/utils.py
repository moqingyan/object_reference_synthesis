import torch 
import json
from cmd_args import cmd_args
import numpy as np
from torch.autograd import Variable
from enum import Enum, unique
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
from itertools import combinations 

AVAILABLE_OBJ_DICT = dict()
AVAILABLE_OBJ_DICT["0"] = ["1", "2"]
AVAILABLE_OBJ_DICT["1"] = ["0", "2"]
AVAILABLE_OBJ_DICT["2"] = ["0", "1"]
ATTRS = ["color", "size", "shape", "material"]
CENTER_RELATION = ["right_center", "behind_center"]

class NodeType(Enum):

    var = 1
    attr = 2
    relation = 3 
    target = 4 
    center_relation = 5
    center_prob = 5
    first = 6 
    second = 7
    gb = 8
    binding = 9
    obj = 10
    prob = 11

class EdgeType(Enum):
    
    edge_attr = 1
    first = 2
    second = 3
    target = 4
    center_relation = 5
    center_prob = 6
    binding = 7
    edge_prob = 8
    gb = 9

# The attributes and edge types are in common over all graphs
class Encoder():

    def __init__(self, config, max_obj_num = cmd_args.max_obj_num,  max_var_num = cmd_args.max_var_num, hidden_dim=cmd_args.hidden_dim):
        self.config = config
        self.node_attrs = []
        self.lookup_list = []
        self.max_obj_num = max_obj_num
        self.max_var_num = max_var_num
    
        self.create_lookup_dict()
        self.embedding = LabelEncoder()
        print(self.lookup_list)
        self.embedding.fit(self.lookup_list)
        

    def create_lookup_dict(self):
        
        for v in self.config["choices"].values():
            self.node_attrs += v

        self.node_attrs.append("right")
        self.node_attrs.append("behind")
        self.node_attrs.append("center_right")
        self.node_attrs.append("center_behind")
        
        self.node_attrs.append("target")
        self.node_attrs.append("first_n")
        self.node_attrs.append("second_n")
        self.node_attrs.append("gb")

        cat = [ f"cat_{c}" for c in range(len(cmd_args.category_offsets))]
        self.node_attrs += cat 

        for ct in range(cmd_args.max_var_num * cmd_args.max_obj_num):
            self.node_attrs.append(f"binding_{ct}")

        for ct in range(cmd_args.max_obj_num * cmd_args.max_obj_num * 2):
            self.node_attrs.append(f"relation_{ct}")

        for ct in range(len(cmd_args.category_offsets) * cmd_args.max_obj_num):
            self.node_attrs.append(f"prob_{ct}")

        for ct in range(cmd_args.max_obj_num * len(self.config["flattened_attr"])):
            self.node_attrs.append(f"attr_{ct}")

        self.node_attrs += ([f"obj_{ct}" for ct in range(self.max_obj_num)])
        self.node_attrs += ([f"var_{ct}" for ct in range(self.max_var_num)])
        
        edges = [f"edge_{ct+1}" for ct in range(len(EdgeType))]
        self.lookup_list = self.node_attrs + self.config["operation_list"] 
        self.edge_offset = len(self.lookup_list)
        self.lookup_list += edges
        self.lookup_list += ["start"]

    def get_embedding(self, attr):
        if not type(attr) == list:
            attr = [attr]
        return (self.embedding.transform(attr))
    
# reward are seperated into: whether a large penalty will be append 
# what is a large penalty in that case 
# whether we are using object left or object eliminated for reward
def policy_gradient_loss(reward_history, prob_history):

    R = 0.0
    rewards = []

    if cmd_args.decay == None:
        decay = 1
    else:
        decay = cmd_args.decay

    # calculate the accumulated reward
    for r in reversed(reward_history):
        rewards.insert(0, R * decay + r)
        R = R * decay + r

    rewards = torch.tensor(rewards)
    # scale the rewards
    if not (len(rewards) == 1) and cmd_args.normalize_reward:
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

    # seems naturally discriminate against longer roll outs, TODO: try averging the loss based on length
    loss = torch.sum( Variable(rewards) * prob_history, -1).mul(-1)
    return loss

# process different types of reward
def get_reward(env):

    reward = 0
    
    if env.is_finished(): 
        return get_final_reward(env)

    if cmd_args.reward_type == "preserved":
        if not env.obj_poss_left[-1] == 0:
            reward =  1 / env.obj_poss_left[-1]
        else:
            reward = 0

    elif cmd_args.reward_type == "eliminated":
        reward = 0
        if (len(env.obj_poss_left) >= 2):
            reward = (env.obj_poss_left[-2] - env.obj_poss_left[-1]) / len(env.graph.scene["objects"])

    elif cmd_args.reward_type == "no_intermediate":
        reward = 0

    elif cmd_args.reward_type == "only_success":
       
        reward = 0
        if (len(env.obj_poss_left) >= 2):
            reward = env.obj_poss_left[-2] - env.obj_poss_left[-1]

    return reward

# Get the final reward
def get_final_reward(env):
    
    if not cmd_args.reward_penalty :
        return 0
    else: 
        if not env.success:
            if cmd_args.reward_type == "preserved": 
                return -1
            elif cmd_args.reward_type == "eliminated":
                return -1
            elif cmd_args.reward_type == "no_intermediate":
                return -1
            elif cmd_args.reward_type == "only_success":
                return -1
        else:
            if cmd_args.reward_type == "preserved": 
                return 1
            elif cmd_args.reward_type == "eliminated":
                return 1
            if cmd_args.reward_type == "no_intermediate":
                return 1
            if cmd_args.reward_type == "only_success":
                return 1

# Generate a configuration dict
def get_config():
    operation_list = ["size", "color", "shape", "material"]
    choices = dict() 

    choices["size"] = ["large", "small"]
    choices["color"] = ["blue", "red", "yellow", "green", "gray", "brown", "purple", "cyan"]
    choices["shape"] = ["cube", "cylinder", "sphere"]
    choices["material"] = ["rubber", "metal"] 

    flattened_attr = []
    for c, v in choices.items():
        for e in v:
            flattened_attr.append(e)

    edge_types = dict()
    edge_types["attributes"] = ["size", "color", "shape", "material"]
    edge_types["spatial_relation"] = ["left", "right", "front", "behind"]
    edge_types["target"] = ["target"]
    edge_types["bonding"] = ["bonding"]

    config = dict()
    config["operation_list"] = operation_list
    config["choices"] = choices
    config["edge_types"] = edge_types
    config["flattened_attr"] = flattened_attr
    return config

def get_operands():
    config = get_config()
    unary = {}
    binary = {}

    for unary_ops in config["choices"].values():
        for unary_op in unary_ops:
            unary[unary_op] = []
    
    binary["right"] = []
    binary["behind"] = []
    
    return unary, binary

def get_all_clauses(config):

    clauses = []

    for op in (config["edge_types"]["attributes"]):
        for attr in config["choices"][op]:
            for var in range(cmd_args.max_var_num):
                clauses.append([op, attr, var])

    for op in ["right", "behind"]:
        for var1 in range(cmd_args.max_var_num):
            for var2 in range(cmd_args.max_var_num):
                if not var1 == var2:
                    clauses.append([op, var1, var2])

    return clauses 

def get_reachable_clauses(clauses, refs):
    reachable = []
    not_reachable = []

    for clause_ct, clause in enumerate(clauses):
        r = False 
        for ref in refs:
            if ref in clause:
                r = True 
        if r:
            reachable.append(clause_ct)
        else:
            not_reachable.append(clause_ct)

    return reachable, not_reachable

def get_reachable_dict(clauses, max_var_num = cmd_args.max_var_num):
    reachable_dict = {}
    unreachable_dict = {}
    refs = []
    for ct in range(max_var_num):
        refs += (list(combinations(range(max_var_num), ct+1)))

    for ref in refs:
        reachable, not_reachable = get_reachable_clauses(clauses, ref)
        reachable_dict[str(list(ref))] = reachable
        unreachable_dict[str(list(ref))] = not_reachable

    return reachable_dict, unreachable_dict

config = get_config()
ACTION_NUM = len(get_all_clauses(config))

class OneHotEmbedding(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.embedding_layer = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        
        encoding = torch.zeros([x.shape[0], self.input_dim])
        encoding = encoding.scatter_(1, x.view(-1,1), 1)
        # print(f"x:{encoding}")
        embedding = self.embedding_layer(encoding)
        return embedding