import torch 
import torch.nn as nn 
import json 
import os
import sys
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

common_path = os.path.abspath(os.path.join(__file__, "../common"))
sys.path.append(common_path)

from scene2graph import Graph, GraphNode, Edge, NodeType, EdgeType
from cmd_args import cmd_args
from utils import AVAILABLE_OBJ_DICT, Encoder
from cmd_args import logging
import copy

class LSTMClauses(nn.Module):

    def __init__(self, clause_decoder):
        super().__init__()
        self.clause_decoder = clause_decoder
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    # one single rollout
    def forward(self, env):
        prob, clause, clause_encoding = self.clause_decoder(env.state)
        env.state = self.gru_cell(cur_state, clause, clause_encoding)
        return prob, clause

# decode a clause we get, or use other method instead
class AttClauseDecoder(nn.Module):

    def __init__(self, encoder, embedding_layer, var_number=cmd_args.max_var_num, hidden_dim=cmd_args.hidden_dim):
        super().__init__()

        config = encoder.config
        # operation_list = config["operation_list"]
        operation_list = config["operation_list"]
        self.choices = config["choices"]
        self.attention_layer = nn.Linear(hidden_dim, 1)
        
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.get_obj1_layer = nn.Sequential(
            nn.Linear(hidden_dim, var_number),
            nn.Softmax(dim=1))
        self.get_obj2_layer = nn.Sequential(
            nn.Linear(hidden_dim, var_number-1),
            nn.Softmax(dim=1))
        self.get_operation_layer =  nn.Sequential(
            nn.Linear(hidden_dim, len(operation_list)),
            nn.Softmax(dim=1))
        self.get_attribute_layers = dict()
        for key, value in self.choices.items():
            self.get_attribute_layers[key] = nn.Sequential(
            nn.Linear(hidden_dim, len(value)),
            nn.Softmax(dim=1))

        self.operation_list = operation_list

        self.state = None
        self.encoder = encoder
        self.embedding_layer = embedding_layer
        self.clause_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.Softmax(dim=1),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), 
            nn.Softmax(dim=1)
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Softmax(dim=1)
        )

    def get_attention(self, embeddings):
        weight = self.attention_layer(embeddings)
        weight = F.softmax(weight, dim=0)
        attention = torch.sum(embeddings * weight, dim=0, keepdim=True)
        return attention
        
    def get_prob(self, layer, eps):
        probs = layer(self.state)
        if not eps == None:
            probs = probs * (1 - eps) + eps / probs.shape[1]
        distr = Categorical(probs)
        
        if cmd_args.test_type == "max" and self.training == False:
             _, sel = probs[0].max(0)
        else:
            sel = distr.sample()
           
        prob = torch.index_select(probs, 1, sel)
        return prob, sel
    
    def get_embedding(self, x):
        x = torch.tensor(self.encoder.get_embedding(x))
        x = self.embedding_layer(x)
        return x

    def get_operator(self, eps):
        
        p_op, selected_operation = self.get_prob(self.get_operation_layer, eps)
        operation = self.operation_list[selected_operation]
        operation_embedding = self.get_embedding(operation)
        return p_op, operation, operation_embedding

    def get_object_1(self, encoder, eps):

        p_o1, selected_object1 = self.get_prob(self.get_obj1_layer, eps)
        object1 = f"var_{int(selected_object1)}"
        object1_embedding = self.get_embedding(object1)
        return p_o1, object1, object1_embedding

    def get_object_2(self, prev, encoder, eps):

        prev_idx = int(prev[-1])
        p_o2, selected_object2 = self.get_prob(self.get_obj2_layer, eps)
        object2 = f"var_{AVAILABLE_OBJ_DICT[str(prev_idx)][int(selected_object2)]}"
        object2_embedding = self.get_embedding(object2)
        return p_o2, object2, object2_embedding

    def get_attr(self, operation, eps):
       
        p_attr, selected_attribute = self.get_prob(self.get_attribute_layers[operation], eps)
        attribute = self.choices[operation][int(selected_attribute)]
        attribute_embedding = self.get_embedding(attribute)
        return p_attr, attribute, attribute_embedding

    def forward(self, graph_embedding, encoder, eps):
        
        self.state = self.get_attention(graph_embedding)
        p_op, operation, operation_embedding = self.get_operator(eps)

        # update_state
        self.state = self.gru_cell(operation_embedding, self.state)
        p_o1, object1, object1_embedding = self.get_object_1(encoder, eps)

        # update_state
        self.state = self.gru_cell(object1_embedding, self.state)
        if operation in ["left", "right", "front", "behind"]:
            p2, c2, e2 = self.get_object_2(object1, encoder, eps)
        else:
            p2, c2, e2 = self.get_attr(operation, eps)

        clause = [operation, c2, object1]
        clause_encoding = clause_layer(torch.stack([operation_embedding, object1_embedding, e2]))

        return (p_op * p_o1 * p2)[0], clause, clause_encoding

if __name__ == "__main__":
    # load the data 
    data_dir = os.path.abspath(__file__ + "../../../data")
    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    # encoder = Encoder(config)

    # construct a mini example
    # decoder = Decoder(encoder)
    decoder = NodeDecoder()