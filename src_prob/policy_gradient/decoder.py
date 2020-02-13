import torch 
import torch.nn as nn 
import json 
import os
import sys
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from cmd_args import cmd_args
from utils import AVAILABLE_OBJ_DICT, Encoder, get_config
from cmd_args import logging
import copy

class NodeDecoder(nn.Module):

    def __init__(self, hidden_dim=cmd_args.hidden_dim):
        super().__init__()
        self.attr_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim, int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), int(cmd_args.hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/4), 1),
            nn.Softmax(dim=0)
        ) 

        self.var1_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim, int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), int(cmd_args.hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/4), 1),
            nn.Softmax(dim=0)
        )

        self.var2_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim, int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), int(cmd_args.hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/4), 1),
            nn.Softmax(dim=0)
        )

        self.rela_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim, int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), int(cmd_args.hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/4), 1),
            nn.Softmax(dim=0)
        )

        self.attr_or_rela_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim, int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), int(cmd_args.hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/4), 1),
            nn.Softmax(dim=0)
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        

    def get_attr_locs(self, graph):
        return graph.get_attr_locs()

    # def get_obj_locs(self, graph):
    #     return graph.objs.values()

    def get_rela_locs(self, graph):
        return graph.get_rela_locs()
    
    # def get_attr_or_rela_locs(self, v, graph):
    #     return graph.get_var_attr_or_rela(v)
    def get_attr_or_rela_locs(self, graph):
        return graph.get_attr_or_rela_locs()

    def get_var_locs(self, graph, rel=None):
        return graph.get_var_locs(rel)
        
    def idx_to_attr(self, graph, sel):
        return graph.get_attr_by_idx(sel)

    def idx_to_rela(self, graph, sel):
        return graph.get_rela_by_idx(sel)

    def idx_to_var(self, graph, sel):
        return graph.get_var_by_idx(sel)

    def idx_to_attr_or_rela(self, graph, sel):
        return graph.get_attr_or_rela_by_idx(sel)

    def locs_to_byte(self, locs, num):
        res = [0] * num
        for i in range(num):
            if i in locs:
                res[i] = 1
        return torch.BoolTensor(res)
    
    def get_prob(self, node_embeddings, graph, locs, layer, phase, eps):
        # graph_embedding is of #node * hidden dim

        node_num = node_embeddings.shape[0]
        locs = self.locs_to_byte(locs, node_num)

        embeddings = node_embeddings[locs]
        if (embeddings.shape[0] == 0):
            print("here!")
        probs = layer(embeddings)
       
        if not eps == None:
            probs = probs * (1 - eps) + eps / probs.shape[0]
        
        # logging.info(f"probs: {probs}")
        distr = Categorical(probs.reshape(probs.shape[0]))
        
        if cmd_args.test_type == "max" and self.training == False:
             _, sel = probs[0].max(0)
        else:
            sel = distr.sample()

        prob = torch.index_select(probs, 0, sel)
        embedding = torch.index_select(embeddings, 0, sel)
        return prob, sel, embedding
 
    def get_attr(self, node_embeddings, graph, phase,  eps):
        locs = self.get_attr_locs(graph)
        prob, select_id, embedding = self.get_prob(node_embeddings, graph, locs, self.attr_score_layer, phase, eps)
        sel_attr = self.idx_to_attr(graph, select_id)
        return prob, embedding, select_id, sel_attr

    def get_var1(self, node_embeddings, graph, ref, phase, eps):
        locs = self.get_var_locs(graph, ref)
        prob, select_id, embedding = self.get_prob(node_embeddings, graph, locs,  self.var1_score_layer, phase, eps)
        var1_loc = locs[select_id]

        sel_var = self.idx_to_var(graph, select_id)
        # print(sel_var)
        return prob, embedding, var1_loc, sel_var

    def get_var2(self, node_embeddings, graph, prev_loc, phase, eps):
        locs = self.get_var_locs(graph)
        locs.remove(prev_loc)

        prob, select_id, embedding = self.get_prob(node_embeddings, graph, locs, self.var2_score_layer, phase, eps)
        
        sel_var = self.idx_to_var(graph, select_id)
        return prob, embedding, select_id, sel_var
                
    def get_rela(self, node_embeddings, graph, phase, eps):
        locs = self.get_rela_locs(graph)
        prob, select_id, embedding = self.get_prob(node_embeddings, graph, locs, self.rela_score_layer, phase, eps)
        sel_rela = self.idx_to_rela(graph, select_id)
        return prob, embedding, select_id, sel_rela
    
    def get_attr_or_rela(self, node_embeddings, graph, phase, eps):
        locs = self.get_attr_or_rela_locs(graph)
        prob, select_id, embedding = self.get_prob(node_embeddings, graph, locs, self.attr_or_rela_score_layer, phase, eps)
        sel_rela_or_attr = self.idx_to_attr_or_rela(graph, select_id)
        return prob, embedding, select_id, sel_rela_or_attr

    def is_attr(self, graph, name):
        return name in graph.attrs

    def get_attr_operation(self, name, configure):
        for key, value in configure["choices"].items():
            if name in value:
                return key 
        return 

    def forward(self, graph_embedding, graph, ref, eps, phase="train"):

        clause = []
        prob = 1

        node_embeddings = graph_embedding[0]
        self.state = self.gru_cell(graph_embedding[1])

        # For testing purpose, only select attribute
        # prob_rela_or_attr, rela_or_attr_embedding, attr_id, sel_rela_or_attr = self.get_attr(node_embeddings, graph, phase, eps)
        prob_rela_or_attr, rela_or_attr_embedding, rela_or_attr_id, sel_rela_or_attr = self.get_attr_or_rela(node_embeddings, graph, phase, eps)
        self.state = self.gru_cell(rela_or_attr_embedding, self.state)
        node_embeddings = torch.stack([self.gru_cell(node_embedding.unsqueeze(0), self.state) for node_embedding in node_embeddings]).reshape(node_embeddings.shape)
        prob *= prob_rela_or_attr

        if self.is_attr(graph, sel_rela_or_attr):
            if not cmd_args.var_space_constraint:
                ref = None

            prob_var1, var1_embedding, var1_loc, sel_var1 = self.get_var1(node_embeddings, graph, ref, phase=phase, eps=eps)
            # print(f"attr: sel_var1, ref: {sel_var1, ref}")
            self.state = self.gru_cell(var1_embedding, self.state)
            node_embeddings = torch.stack([self.gru_cell(node_embedding.unsqueeze(0), self.state) for node_embedding in node_embeddings]).reshape(node_embeddings.shape)
            prob *= prob_var1
            
            operation = self.get_attr_operation(sel_rela_or_attr, graph.config)
            clause.append(operation)
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
        else:

            prob_var1, var1_embedding, var1_loc, sel_var1 = self.get_var1(node_embeddings, graph, ref=None, phase=phase, eps=eps)
            # print(f"rela: sel_var1, ref: {sel_var1, ref}")
            self.state = self.gru_cell(var1_embedding, self.state)
            node_embeddings = torch.stack([self.gru_cell(node_embedding.unsqueeze(0), self.state) for node_embedding in node_embeddings]).reshape(node_embeddings.shape)
            prob *= prob_var1

            prob_var2, var2_embedding, var2_id, sel_var2 = self.get_var2(node_embeddings, graph, var1_loc, phase, eps)
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
            clause.append(sel_var2)
            prob *= prob_var2

        return prob[0], clause

# decode a clause we get, or use other method instead
class GlobalDecoder(nn.Module):

    def __init__(self, encoder, embedding_layer, var_number=cmd_args.max_var_num, hidden_dim=cmd_args.hidden_dim):
        super().__init__()

        config = encoder.config
        operation_list = config["operation_list"]
        self.choices = config["choices"]

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

    def get_prob(self, layer, phase, eps):
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

    def get_operator(self, phase, eps):
        
        p_op, selected_operation = self.get_prob(self.get_operation_layer, phase, eps)
        operation = self.operation_list[selected_operation]
        operation_embedding = self.get_embedding(operation)
        return p_op, operation, operation_embedding

    def get_object_1(self, phase, encoder, eps):

        p_o1, selected_object1 = self.get_prob(self.get_obj1_layer, phase, eps)
        object1 = f"var_{int(selected_object1)}"
        object1_embedding = self.get_embedding(object1)
        return p_o1, object1, object1_embedding

    def get_object_2(self, phase, prev, encoder, eps):

        prev_idx = int(prev[-1])
        p_o2, selected_object2 = self.get_prob(self.get_obj2_layer, phase, eps)
        object2 = f"var_{AVAILABLE_OBJ_DICT[str(prev_idx)][int(selected_object2)]}"
        object2_embedding = self.get_embedding(object2)
        return p_o2, object2, object2_embedding

    def get_attr(self, operation, phase, eps):
       
        p_attr, selected_attribute = self.get_prob(self.get_attribute_layers[operation], phase, eps)
        attribute = self.choices[operation][int(selected_attribute)]
        attribute_embedding = self.get_embedding(attribute)
        return p_attr, attribute, attribute_embedding

    def forward(self, graph_embedding, encoder, eps, phase="train"):

        self.state = global_embedding
        p_op, operation, operation_embedding = self.get_operator(phase, eps)

        # update_state
        self.state = self.gru_cell(operation_embedding, self.state)
        p_o1, object1, object1_embedding = self.get_object_1(phase, encoder, eps)

        # update_state
        self.state = self.gru_cell(object1_embedding, self.state)
        if operation in ["left", "right", "front", "behind"]:
            p2, c2, e2 = self.get_object_2(phase, object1, encoder, eps)
        else:
            p2, c2, e2 = self.get_attr(operation, phase, eps)

        clause = [operation, c2, object1]
        # logging.info(f"probs of clause parts: {(p_op, p_o1, p2)}, total prob: {p_op * p_o1 * p2}, state ori: {graph_embedding}, state updated: {self.state}")
        return (p_op * p_o1 * p2)[0], clause

        
# decode a clause we get, or use other method instead
class AttDecoder(nn.Module):

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

    def get_attention(self, embeddings):
        weight = self.attention_layer(embeddings)
        weight = F.softmax(weight, dim=0)
        attention = torch.sum(embeddings * weight, dim=0, keepdim=True)
        return attention
        
    def get_prob(self, layer, phase, eps):
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

    def get_operator(self, phase, eps):
        
        p_op, selected_operation = self.get_prob(self.get_operation_layer, phase, eps)
        operation = self.operation_list[selected_operation]
        operation_embedding = self.get_embedding(operation)
        return p_op, operation, operation_embedding

    def get_object_1(self, phase, encoder, eps):

        p_o1, selected_object1 = self.get_prob(self.get_obj1_layer, phase, eps)
        object1 = f"var_{int(selected_object1)}"
        object1_embedding = self.get_embedding(object1)
        return p_o1, object1, object1_embedding

    def get_object_2(self, phase, prev, encoder, eps):

        prev_idx = int(prev[-1])
        p_o2, selected_object2 = self.get_prob(self.get_obj2_layer, phase, eps)
        object2 = f"var_{AVAILABLE_OBJ_DICT[str(prev_idx)][int(selected_object2)]}"
        object2_embedding = self.get_embedding(object2)
        return p_o2, object2, object2_embedding

    def get_attr(self, operation, phase, eps):
       
        p_attr, selected_attribute = self.get_prob(self.get_attribute_layers[operation], phase, eps)
        attribute = self.choices[operation][int(selected_attribute)]
        attribute_embedding = self.get_embedding(attribute)
        return p_attr, attribute, attribute_embedding

    def forward(self, graph_embedding, encoder, eps, phase="train"):
        
        self.state = self.get_attention(graph_embedding)
        p_op, operation, operation_embedding = self.get_operator(phase, eps)

        # update_state
        self.state = self.gru_cell(operation_embedding, self.state)
        p_o1, object1, object1_embedding = self.get_object_1(phase, encoder, eps)

        # update_state
        self.state = self.gru_cell(object1_embedding, self.state)
        if operation in ["left", "right", "front", "behind"]:
            p2, c2, e2 = self.get_object_2(phase, object1, encoder, eps)
        else:
            p2, c2, e2 = self.get_attr(operation, phase, eps)

        clause = [operation, c2, object1]
        # logging.info(f"probs of clause parts: {(p_op, p_o1, p2)}, total prob: {p_op * p_o1 * p2}, state ori: {graph_embedding}, state updated: {self.state}")
        return (p_op * p_o1 * p2)[0], clause

if __name__ == "__main__":
    # load the data
    config = get_config()
    
    # construct a mini example
    # decoder = Decoder(encoder)
    decoder = NodeDecoder()