import torch 
import torch.nn as nn 
import json 
import os
import sys
from torch.distributions.categorical import Categorical
from torch_geometric.data import DataLoader
import torch.nn.functional as F

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from scene2graph import Graph, GraphNode, Edge, NodeType, EdgeType
from cmd_args import cmd_args
from utils import AVAILABLE_OBJ_DICT, Encoder, get_all_clauses
from cmd_args import logging
import copy

from embedding import GNNGL, SceneDataset, create_dataset
from env import Env 

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

    # def idx_to_obj(self, graph, sel):
    #     return graph.objs.keys()[sel]

    def idx_to_rela(self, graph, sel):
        return graph.get_rela_by_idx(sel)

    def idx_to_var(self, graph, sel):
        return graph.get_var_by_idx(sel)

    # def idx_to_attr_or_rela(self, graph, sel):
    #     return graph.nodes[sel].name 
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
        embedding = torch.index_select(embeddings, 0, sel)[:, :cmd_args.hidden_dim]
        return prob, sel, embedding

    def get_probs(self, node_embeddings, graph, locs, layer, phase, eps):
        # graph_embedding is of #node * hidden dim

        node_num = node_embeddings.shape[0]
        locs = self.locs_to_byte(locs, node_num)

        embeddings = node_embeddings[locs]
        if (embeddings.shape[0] == 0):
            print("here!")
        probs = layer(embeddings)
       
        if not eps == None:
            probs = probs * (1 - eps) + eps / probs.shape[0]
        
        return probs, embeddings
 
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

    # def get_attr_or_rela(self, v, node_embeddings, graph, phase, eps):
    #     locs = self.get_attr_or_rela_locs(v, graph)
    #     prob, select_id, embedding = self.get_prob(node_embeddings, graph, locs, self.attr_or_rela_score_layer, phase, eps)
    #     # sel_rela_or_attr = self.idx_to_attr_or_rela(graph, select_id)
    #     sel_rela_or_attr = self.idx_to_attr_or_rela(graph, locs[select_id])
    #     return prob, embedding, select_id, sel_rela_or_attr
    
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

    def unroll(self, graph_embedding, graph, ref, eps, phase="train"):
        clauses = []
        probs = []

        local_embeddings = graph_embedding[0]
        global_embedding = graph_embedding[1]

        self.state = global_embedding
        node_embeddings = torch.stack([torch.cat((local_embedding, self.state.squeeze(0))) for local_embedding in local_embeddings])

        var1_locs = self.get_var_locs(graph, ref)
        # var1_names = graph.vars.keys()
        var1_probs, var1_embeddings = self.get_probs(node_embeddings, graph, var1_locs, self.var1_score_layer, phase, eps)

        var1_states = [ self.gru_cell(var1_embedding.reshape(1, -1)[:, :cmd_args.hidden_dim], self.state) for var1_embedding in var1_embeddings ]

        for var1_id, (var1_prob, current_state) in enumerate(zip(var1_probs, var1_states)):
            node_embeddings = torch.stack([torch.cat((local_embedding, current_state.squeeze(0))) for local_embedding in local_embeddings])
            var1_name = graph.nodes[var1_locs[var1_id]].name 
            attr_or_rela_locs = self.get_attr_or_rela_locs(graph)
            attr_or_rela_names = graph.attr_or_rela.keys()
            attr_or_rela_probs, attr_or_rela_embeddings = self.get_probs(node_embeddings, graph, attr_or_rela_locs, self.attr_or_rela_score_layer, phase, eps)
            
            prob_or_rela_states = [self.gru_cell(attr_or_rela_embedding.reshape(1, -1)[:, :cmd_args.hidden_dim], self.state) for attr_or_rela_embedding in attr_or_rela_embeddings]
            
            for attr_or_rela_id, (attr_or_rela_prob, prob_or_rela_state) in enumerate(zip(attr_or_rela_probs, prob_or_rela_states)):
                node_embeddings = torch.stack([torch.cat((local_embedding, prob_or_rela_state.squeeze(0))) for local_embedding in local_embeddings])
                sel_rela_or_attr = list(attr_or_rela_names)[attr_or_rela_id]

                if self.is_attr(graph, sel_rela_or_attr):
                    
                    operation = self.get_attr_operation(sel_rela_or_attr, graph.config)
                    clauses.append([operation, sel_rela_or_attr, var1_name])
                    probs.append(var1_prob * attr_or_rela_prob)

                else:
                    var2_locs = self.get_var_locs(graph)
                    var2_locs.remove(var1_locs[var1_id])
                    var2_probs, _ = self.get_probs(node_embeddings, graph, var2_locs, self.var2_score_layer, phase, eps)
                    var2_names = [graph.nodes[select_id].name for select_id in var2_locs]

                    for var2_name, var2_prob in zip(var2_names, var2_probs):
                        clauses.append([sel_rela_or_attr, var1_name, var2_name])
                        probs.append(var1_prob * attr_or_rela_prob * var2_prob)

        return probs, clauses

    def forward(self, graph_embedding, graph, ref, eps, phase="train"):
        
        # # For uniform distribution among clauses
        # clause = []
        # prob = 1

        # node_embeddings = graph_embedding[0]
        # self.state = self.gru_cell(graph_embedding[1])

        # prob_var1, var1_embedding, var1_id, sel_var1 = self.get_var1(node_embeddings, graph, phase, eps)
        # self.state = self.gru_cell(var1_embedding, self.state)
        # node_embeddings = torch.stack([self.gru_cell(node_embedding.unsqueeze(0), self.state) for node_embedding in node_embeddings]).reshape(node_embeddings.shape)
        # prob *= prob_var1
        
        # if cmd_args.max_var_num == 1:
        #     prob_rela_or_attr, rela_or_attr_embedding, attr_id, sel_rela_or_attr = self.get_attr(node_embeddings, graph, phase, eps)
        # else:
        #     prob_rela_or_attr, rela_or_attr_embedding, rela_or_attr_id, sel_rela_or_attr = self.get_attr_or_rela( graph.vars[var1_id.item()], node_embeddings, graph, phase, eps)
        
        # self.state = self.gru_cell(rela_or_attr_embedding, self.state)
        # node_embeddings = torch.stack([self.gru_cell(node_embedding.unsqueeze(0), self.state) for node_embedding in node_embeddings]).reshape(node_embeddings.shape)
        # prob *= prob_rela_or_attr
        
        
        # # TODO: we can also compare the performance against the original graph embedding
        # # self.state = self.gru_cell(rela_or_attr_embedding , self.state)
        # # node_embeddings = torch.stack([self.gru_cell(node_embedding.unsqueeze(0), self.state) for node_embedding in graph_embedding]).reshape(graph_embedding.shape)
        # print(sel_var1)
        # print(sel_rela_or_attr)

        # # redundant var2 for more uniformed probs
        # prob_var2, var2_embedding, var2_id, sel_var2 = self.get_var2(node_embeddings, graph, var1_id, phase, eps)
        # prob *= prob_var2

        # if self.is_attr(graph, sel_rela_or_attr):
        #     operation = self.get_attr_operation(sel_rela_or_attr, graph.config)
        #     clause.append(operation)
        #     clause.append(sel_rela_or_attr)
        #     clause.append(sel_var1)
        # else:
        #     clause.append(sel_rela_or_attr)
        #     clause.append(sel_var1)
        #     clause.append(sel_var2)

        # print(clause)
        # return prob[0], clause
        
        clause = []
        prob = 1

        local_embeddings = graph_embedding[0]
        if cmd_args.global_node:
            global_id = graph.global_idx
            global_embedding = torch.index_select(local_embeddings, 1, global_id) # double check dim
        else:
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
        
        # TODO: we can also compare the performance against the original graph embedding
        # self.state = self.gru_cell(rela_or_attr_embedding , self.state)
        # node_embeddings = torch.stack([self.gru_cell(node_embedding.unsqueeze(0), self.state) for node_embedding in graph_embedding]).reshape(graph_embedding.shape)

        if self.is_attr(graph, sel_rela_or_attr):
            
            operation = self.get_attr_operation(sel_rela_or_attr, graph.config)
            clause.append(operation)
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
        else:

            prob_var2, var2_embedding, var2_id, sel_var2 = self.get_var2(node_embeddings, graph, var1_loc, phase, eps)
            clause.append(sel_rela_or_attr)
            clause.append(sel_var1)
            clause.append(sel_var2)
            prob *= prob_var2

        return prob[0], clause

# decode a clause we get, or use other method instead
class GlobalDecoder2(nn.Module):

    def __init__(self, encoder, embedding_layer, var_number=cmd_args.max_var_num, hidden_dim=cmd_args.hidden_dim):
        super().__init__()

        config = encoder.config
        operation_list = config["operation_list"]
        self.choices = config["choices"]

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

    def forward(self, graph_embedding, graph, encoder, eps, phase="train"):

        if cmd_args.gnn_version == "GNNGlobal":
            self.state = graph_embedding
        elif cmd_args.gnn_version == "GNNGL":
            local_embeddings = graph_embedding[0]
            if cmd_args.global_node:
                global_id = graph.global_idx
                global_embedding = local_embeddings[-1].reshape([1, cmd_args.hidden_dim])
            else:
                global_embedding = graph_embedding[1]
            self.state = global_embedding
        else:
            raise "decoder not support this gnn version"
        
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
class GlobalDecoder(nn.Module):

    def __init__(self, encoder, embedding_layer, hidden_dim=cmd_args.hidden_dim):
        super().__init__()

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.encoder = encoder
        self.actions = get_all_clauses(encoder.config)
        self.embedding_layer = embedding_layer
        
        self.prob_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1))

    def forward(self, graph_embeddings, graph, eps):
        
        if cmd_args.gnn_version == "GNNGlobal":
            self.state = graph_embeddings
        elif cmd_args.gnn_version == "GNNGL":
            local_embeddings = graph_embeddings[0]
            if cmd_args.global_node:
                global_id = graph.global_idx
                global_embedding = local_embeddings[-1].reshape([1, cmd_args.hidden_dim])
            else:
                global_embedding = graph_embeddings[1]
            self.state = global_embedding
        else:
            raise "decoder not support this gnn version"

        clause_embeddings = []

        for clause in self.actions:

            if clause[0] == "behind" or clause[0] == "right":
                embeddings = torch.tensor((self.encoder.get_embedding(clause[0]), 
                              self.encoder.get_embedding(f"var_{clause[1]}"), 
                              self.encoder.get_embedding(f"var_{clause[2]}")))
                embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[-1])
                embeddings = self.embedding_layer(embeddings)

                rela_embedding = embeddings[0]
                v1_embedding =   embeddings[1]
                v2_embedding =   embeddings[2]

                s1 = self.gru_cell(rela_embedding.view(1, rela_embedding.shape[-1]), self.state)
                s2 = self.gru_cell(v1_embedding.view(1, v1_embedding.shape[-1]), s1)
                s3 = self.gru_cell(v2_embedding.view(1, v2_embedding.shape[-1]), s2)
                clause_embeddings.append(s3)
                
            else:
                embeddings = torch.tensor((self.encoder.get_embedding(clause[1]), 
                                           self.encoder.get_embedding(f"var_{clause[2]}")))
                embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[-1])
                embeddings = self.embedding_layer(embeddings)

                attr_embedding = embeddings[0]
                var_embedding =  embeddings[1]
                
                s1 = self.gru_cell(attr_embedding.view(1, attr_embedding.shape[-1]), self.state)
                s2 = self.gru_cell(var_embedding.view(1, var_embedding.shape[-1]), s1)

                clause_embeddings.append(s2)

        clause_embeddings = torch.cat(clause_embeddings)
        probs = self.prob_layer(clause_embeddings).view(1, -1)
        
        if not eps == None:
            probs = probs * (1 - eps) + eps / probs.shape[1]
        distr = Categorical(probs)
        
        if cmd_args.test_type == "max" and self.training == False:
             _, sel = probs[0].max(0)
        else:
            sel = distr.sample()
            
        prob = torch.index_select(probs, 1, sel)

        return prob[0], self.actions[sel]

# decode a clause we get, or use other method instead
class AttDecoder(nn.Module):

    def __init__(self, encoder, embedding_layer, var_number=cmd_args.max_var_num, hidden_dim=cmd_args.hidden_dim):
        super().__init__()

        config = encoder.config
        # operation_list = config["operation_list"]
        operation_list = config["operation_list"]
        self.choices = config["choices"]
        self.hidden_dim = hidden_dim
        
        self.gru_cell = nn.GRU(hidden_dim, hidden_dim)
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
        self.op_attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, cmd_args.max_node_num))
        self.ob1_attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, cmd_args.max_node_num))
        self.start_attention_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2), 
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, cmd_args.max_node_num))
        self.attn_combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # self.state = None
        self.encoder = encoder
        self.embedding_layer = embedding_layer

    def get_attention(self, input, hidden, embeddings, attention_layer):
        attn_weights = F.softmax(attention_layer(torch.cat((hidden[0], input[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights[:, :embeddings.shape[0]].unsqueeze(0),
                                embeddings.unsqueeze(0))
        output = torch.cat((input[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru_cell(output, hidden)

        return output, hidden
        
    def get_prob(self, state, layer, phase, eps):
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
    
    def get_embedding(self, i):
        
        x = torch.tensor(self.encoder.get_embedding(i))
        x = self.embedding_layer(x)
        # if (i == "start"):
        #     print(f"x:{x}")
        return x

    def get_operator(self, state, phase, eps):
        
        p_op, selected_operation = self.get_prob(state, self.get_operation_layer, phase, eps)
        operation = self.operation_list[selected_operation]
        operation_embedding = self.get_embedding(operation)
        return p_op, operation, operation_embedding

    def get_object_1(self, state, phase, eps):

        p_o1, selected_object1 = self.get_prob(state, self.get_obj1_layer, phase, eps)
        object1 = f"var_{int(selected_object1)}"
        object1_embedding = self.get_embedding(object1)
        return p_o1, object1, object1_embedding

    def get_object_2(self, state, phase, prev, eps):

        prev_idx = int(prev[-1])
        p_o2, selected_object2 = self.get_prob(state, self.get_obj2_layer, phase, eps)
        object2 = f"var_{AVAILABLE_OBJ_DICT[str(prev_idx)][int(selected_object2)]}"
        object2_embedding = self.get_embedding(object2)
        return p_o2, object2, object2_embedding

    def get_attr(self, state, operation, phase, eps):
       
        p_attr, selected_attribute = self.get_prob(state, self.get_attribute_layers[operation], phase, eps)
        attribute = self.choices[operation][int(selected_attribute)]
        attribute_embedding = self.get_embedding(attribute)
        return p_attr, attribute, attribute_embedding

    def forward(self, graph_embedding, eps, phase="train"):
        
        if cmd_args.gnn_version == "GNNLocal":
            embeddings = graph_embedding
        # elif cmd_args.gnn_version == "GNNGL":
        #     local_embeddings = graph_embedding[0]
        #     if cmd_args.global_node:
        #         global_id = graph.global_idx
        #         global_embedding = local_embeddings[-1].reshape([1, cmd_args.hidden_dim])
        #     else:
        #         global_embedding = graph_embedding[1]
        #     self.state = global_embedding
        else:
            raise Exception("decoder not support this gnn version")

        hidden = torch.zeros(1, 1, self.hidden_dim)
        start = self.get_embedding("start").view(1, 1, -1)
        
        att, hidden = self.get_attention(start, hidden, embeddings, self.start_attention_layer)
        p_op, operation, operation_embedding = self.get_operator(att[0], phase, eps)

        att, hidden = self.get_attention(operation_embedding.view(1, 1, -1), hidden, embeddings, self.op_attention_layer)

        p_o1, object1, object1_embedding = self.get_object_1(att[0], phase, eps)
        att, hidden = self.get_attention(object1_embedding.view(1, 1, -1), hidden, embeddings, self.ob1_attention_layer)

        if operation in ["left", "right", "front", "behind"]:
            p2, c2, e2 = self.get_object_2(att[0], phase, object1, eps)
        else:
            p2, c2, e2 = self.get_attr(att[0], operation, phase, eps)

        clause = [operation, c2, object1]
        # logging.info(f"probs of clause parts: {(p_op, p_o1, p2)}, total prob: {p_op * p_o1 * p2}, state ori: {graph_embedding}, state updated: {self.state}")
        return (p_op * p_o1 * p2)[0], clause

if __name__ == "__main__":
    # load the data 
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    
    model_dir = os.path.abspath(os.path.join(data_dir, "oak_model/model"))
    cont_res_dir = os.path.abspath(os.path.join(data_dir, "eval_result/cont_res_things_15"))

    scene_file_name = "7_things_3_same.json"
    graph_file_name = "7_things_3_same.pkl"
    dataset_name = "7_things_3_same.pt"

    cmd_args.graph_file_name = graph_file_name
    cmd_args.scene_file_name = scene_file_name
    cmd_args.dataset_name = dataset_name
    print(cmd_args)

    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, scene_file_name))
    graphs_path = os.path.join(raw_path, graph_file_name)

    graphs, scene_dataset = create_dataset(data_dir, scenes_path, graphs_path)
    embedding_layer = nn.Embedding(len(scene_dataset.attr_encoder.lookup_list), cmd_args.hidden_dim)
    gnn = GNNGL(scene_dataset, embedding_layer)

    # --- Finished Load dataset , construct decoder ---- #
    decoder = NodeDecoder()
    ref = [0, 1]

    dataloader = DataLoader(scene_dataset)
    for data_point in dataloader:
        graph = graphs[data_point.graph_id]
        env = Env(data_point, graph, config, scene_dataset.attr_encoder)
        graph_embedding = gnn(data_point)
        probs, clauses = decoder.unroll(graph_embedding, graph, ref, eps=0)
