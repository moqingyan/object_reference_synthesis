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
from utils import AVAILABLE_OBJ_DICT, Encoder, get_all_clauses
from cmd_args import logging
import copy

# class LSTMClauses(nn.Module):

#     def __init__(self, clause_decoder):
#         super().__init__()
#         self.clause_decoder = clause_decoder
#         self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

#     # one single rollout
#     def forward(self, env):
#         prob, clause, clause_encoding = self.clause_decoder(env.state)
#         env.state = self.gru_cell(cur_state, clause, clause_encoding)
#         return prob, clause

def get_clauses_idx(graph, actions):

    names = graph.get_nodes()
    name_dict = {}
    for name_id, name in enumerate(names):
        name_dict[name] = name_id 
    
    unary_clauses_idx = []
    binary_clauses_idx = []

    for clause in actions:
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

class ClauseDecoder(nn.Module):

    def __init__(self, hidden_dim=cmd_args.hidden_dim):
        super().__init__()
        self.binary_op_layer = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        self.ternary_op_layer = nn.Sequential(
            nn.Linear(hidden_dim*4, hidden_dim*2),
            nn.ReLU(),
            nn.Linear(hidden_dim*2, hidden_dim)
        )

        self.common_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def get_var_locs(self, graph, rel=None):
        return graph.get_var_locs(rel)

    def idx_to_attr(self, graph, sel):
        return graph.get_attr_by_idx(sel)

    def idx_to_rela(self, graph, sel):
        return graph.get_rela_by_idx(sel)

    def idx_to_var(self, graph, sel):
        return int(graph.get_var_by_idx(sel)[-1])

    def idx_to_attr_or_rela(self, graph, sel):
        return graph.get_attr_or_rela_by_idx(sel)

    def locs_to_byte(self, locs, num):
        res = [0] * num
        for i in range(num):
            if i in locs:
                res[i] = 1
        return torch.BoolTensor(res)

    def forward(self, graph_embeddings, graph, ref, eps, phase="train"):

        local_embedding, global_embedding = graph_embeddings
        actions = get_all_clauses(graph.config)
        unary_clauses_idx, binary_clauses_idx = get_clauses_idx(graph, actions)
        x = []
        
        global_embedding_exp = global_embedding.expand(len(unary_clauses_idx), 1, global_embedding.shape[1])
        unary_rep = torch.cat((local_embedding[unary_clauses_idx, :], global_embedding_exp), dim=1).view(len(unary_clauses_idx), 3 * global_embedding.shape[-1])
        x.append(self.binary_op_layer(unary_rep))

        global_embedding_exp = global_embedding.expand(len(binary_clauses_idx), 1, global_embedding.shape[1])
        binary_rep = torch.cat((local_embedding[binary_clauses_idx, :], global_embedding_exp), dim=1).view(len(binary_clauses_idx), 4 * global_embedding.shape[-1])
        x.append(self.ternary_op_layer(binary_rep))
        # logging.info(f"local embedding: {local_embedding}")
        
        x = torch.cat(x)
        probs = F.softmax(self.common_layer(x).view(-1))

        distr = Categorical(probs)
        if cmd_args.test_type == "max" and self.training == False:
             _, sel = probs[0].max(0)
        else:
            sel = distr.sample()
           
        prob = torch.index_select(probs, 0, sel)
        clause = actions[sel]
        clause_embedding = x[sel]
        local_embedding = self.gru_cell(clause_embedding.expand(local_embedding.shape[0], clause_embedding.shape[-1]), local_embedding)
        global_embedding = self.gru_cell(clause_embedding.view(1, clause_embedding.shape[-1]), global_embedding)

        return prob, clause, (local_embedding, global_embedding)

# decode a clause we get, or use other method instead
class AttClauseDecoder(nn.Module):

    def __init__(self, encoder, embedding_layer, var_number=cmd_args.max_var_num, hidden_dim=cmd_args.hidden_dim):
        super().__init__()

        config = encoder.config
        operation_list = config["operation_list"]
        self.choices = config["choices"]
        self.attention_layer = nn.Linear(hidden_dim, hidden_dim)
        
        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.get_obj1_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, var_number),
            nn.Softmax(dim=1))
        self.get_obj2_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, var_number-1),
            nn.Softmax(dim=1))
        self.get_operation_layer =  nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(operation_list)),
            nn.Softmax(dim=1))
        self.get_attribute_layers = dict()
        for key, value in self.choices.items():
            self.get_attribute_layers[key] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(value)),
                nn.Softmax(dim=1))

        self.operation_list = operation_list

        self.state = None
        self.encoder = encoder
        self.embedding_layer = embedding_layer
        # self.clause_layer = nn.Sequential(
        #     nn.Linear(hidden_dim * 3, hidden_dim * 2),
        #     nn.Softmax(dim=1),
        #     nn.Linear(hidden_dim * 2, hidden_dim * 2),  
        #     nn.Softmax(dim=1)
        #     nn.Linear(hidden_dim * 2, hidden_dim),
        #     nn.Softmax(dim=1)
        # )

    def get_attention(self, embeddings):
        weight = self.attention_layer(embeddings)
        weight = F.softmax(weight, dim=1)
        attention = embeddings * weight
        return attention
        
    def get_prob(self, state, layer, eps):
        probs = layer(state)
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

    def get_operator(self, state, eps):
        
        p_op, selected_operation = self.get_prob(state, self.get_operation_layer, eps)
        operation = self.operation_list[selected_operation]
        operation_embedding = self.get_embedding(operation)
        return p_op, operation, operation_embedding

    def get_object_1(self, state, encoder, eps):

        p_o1, selected_object1 = self.get_prob(state, self.get_obj1_layer, eps)
        object1 = f"var_{int(selected_object1)}"
        object1_embedding = self.get_embedding(object1)
        return p_o1, object1, object1_embedding

    def get_object_2(self, state, prev, encoder, eps):

        prev_idx = int(prev[-1])
        p_o2, selected_object2 = self.get_prob(state, self.get_obj2_layer, eps)
        object2 = f"var_{AVAILABLE_OBJ_DICT[str(prev_idx)][int(selected_object2)]}"
        object2_embedding = self.get_embedding(object2)
        return p_o2, object2, object2_embedding

    def get_attr(self, state, operation, eps):
       
        p_attr, selected_attribute = self.get_prob(state, self.get_attribute_layers[operation], eps)
        attribute = self.choices[operation][int(selected_attribute)]
        attribute_embedding = self.get_embedding(attribute)
        return p_attr, attribute, attribute_embedding

    def forward(self, env_state, encoder, eps):
        
        att_state = self.get_attention(env_state)
        p_op, operation, operation_embedding = self.get_operator(att_state, eps)

        # update_state
        next_state = self.gru_cell(operation_embedding, env_state)
        att_state = self.get_attention(next_state)
        p_o1, object1, object1_embedding = self.get_object_1(att_state, encoder, eps)

        # update_state
        next_state = self.gru_cell(object1_embedding, next_state)
        att_state = self.get_attention(next_state)
        if operation in ["left", "right", "front", "behind"]:
            p2, c2, e2 = self.get_object_2(att_state, object1, encoder, eps)
        else:
            p2, c2, e2 = self.get_attr(att_state, operation, eps)

        next_state = self.gru_cell(e2, next_state)
        clause = [operation, c2, object1]
        # clause_encoding = clause_layer(torch.stack([operation_embedding, object1_embedding, e2]))

        return (p_op * p_o1 * p2)[0], clause, next_state

class NodeDecoder(nn.Module):

    def __init__(self, hidden_dim=cmd_args.hidden_dim):
        super().__init__()
        self.attr_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
            nn.Softmax(dim=0)
        ) 

        self.var1_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
            nn.Softmax(dim=0)
        )

        self.var2_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
            nn.Softmax(dim=0)
        )

        self.rela_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
            nn.Softmax(dim=0)
        )

        self.attr_or_rela_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
            nn.Softmax(dim=0)
        )

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)

    def get_attr_locs(self, graph):
        return graph.attrs.values()

    # def get_obj_locs(self, graph):
    #     return graph.objs.values()

    def get_rela_locs(self, graph):
        return graph.rela_center.values()
    
    # def get_attr_or_rela_locs(self, v, graph):
    #     return graph.get_var_attr_or_rela(v)
    def get_attr_or_rela_locs(self, graph):
        return graph.attr_or_rela.values()

    def get_var_locs(self, graph, rel=None):
        if not (type(rel) == type(None)):
            locs = []
            for r in rel:
                locs.append(graph.vars[r])
            return locs
        else:
            return list(graph.vars.values())


    def idx_to_attr(self, graph, sel):
        return list(graph.attrs.keys())[sel]

    # def idx_to_obj(self, graph, sel):
    #     return graph.objs.keys()[sel]

    def idx_to_rela(self, graph, sel):
        return list(graph.rela_center.keys())[sel]

    def idx_to_var(self, graph, sel):
        return list(graph.vars.keys())[sel]

    # def idx_to_attr_or_rela(self, graph, sel):
    #     return graph.nodes[sel].name 
    def idx_to_attr_or_rela(self, graph, sel):
        return list(graph.attr_or_rela.keys())[sel]

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
        embedding = torch.index_select(embeddings, 0, sel)[:, :cmd_args.hidden_dim]
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

        sel_var = graph.nodes[locs[select_id]].name
        # print(sel_var)
        return prob, embedding, var1_loc, sel_var

    def get_var2(self, node_embeddings, graph, prev_loc, phase, eps):
        locs = self.get_var_locs(graph)
        # print(f"locs: {locs}")
        locs.remove(prev_loc)
        # print(f"prev_loc: {prev_loc}")

        prob, select_id, embedding = self.get_prob(node_embeddings, graph, locs, self.var2_score_layer, phase, eps)
        
        sel_var = graph.nodes[locs[select_id]].name
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
        return name in graph.attrs.keys()

    def get_attr_operation(self, name, configure):
        for key, value in configure["choices"].items():
            if name in value:
                return key 
        return 

    def forward(self, graph_embedding, graph, eps, phase="train"):
        
        clause = []
        prob = 1

        local_embeddings = graph_embedding[0]
        global_embedding = graph_embedding[1]

        self.state = global_embedding
        node_embeddings = torch.stack([torch.cat((local_embedding, self.state.squeeze(0))) for local_embedding in local_embeddings])

        prob_var1, var1_embedding, var1_loc, sel_var1 = self.get_var1(node_embeddings, graph, ref=None, phase=phase, eps=eps)
        self.state = self.gru_cell(var1_embedding, self.state)
        node_embeddings = torch.stack([torch.cat((local_embedding, self.state.squeeze(0))) for local_embedding in local_embeddings])
        prob *= prob_var1
        
        prob_rela_or_attr, rela_or_attr_embedding, rela_or_attr_id, sel_rela_or_attr = self.get_attr_or_rela(node_embeddings, graph, phase, eps)
        self.state = self.gru_cell(rela_or_attr_embedding, self.state)
        node_embeddings = torch.stack([torch.cat((local_embedding, self.state.squeeze(0))) for local_embedding in local_embeddings])
        prob *= prob_rela_or_attr
        
        if self.is_attr(graph, sel_rela_or_attr):
            
            operation = self.get_attr_operation(sel_rela_or_attr, graph.config)
            clause.append(operation)
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
        else:

            prob_var2, var2_embedding, var2_id, sel_var2 = self.get_var2(node_embeddings, graph, var1_loc, phase, eps)
            self.state = self.gru_cell(var2_embedding, self.state)
            
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
            clause.append(sel_var2)
            prob *= prob_var2

        return prob[0], clause, (local_embeddings, self.state)


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