from collections import namedtuple
import torch 
import torch.nn as nn 
import os 
import sys
import random
import math
import torch.nn.functional as F

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from cmd_args import cmd_args, logging
from embedding import GNN, SceneDataset, GNNLocal, GNNGL, GNNGlobal, graph2data
from utils import OneHotEmbedding
import torch.nn as nn

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

gnn_models = {}
gnn_models["GNNLocal"] = GNNLocal
gnn_models["GNNGL"] = GNNGL
gnn_models["GNNGlobal"] = GNNGlobal


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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

    def forward(self, graph_embeddings, state):
        local_embedding, global_embedding = graph_embeddings
        unary_clauses_idx, binary_clauses_idx = state.get_clauses_idx()

        x = []
        
        global_embedding_exp = global_embedding.expand(len(unary_clauses_idx), 1, global_embedding.shape[1])
        unary_rep = torch.cat((local_embedding[unary_clauses_idx, :], global_embedding_exp), dim=1).view(len(unary_clauses_idx), 3 * global_embedding.shape[-1])
        x.append(self.common_layer(self.binary_op_layer(unary_rep)).view(-1))

        global_embedding_exp = global_embedding.expand(len(binary_clauses_idx), 1, global_embedding.shape[1])
        binary_rep = torch.cat((local_embedding[binary_clauses_idx, :], global_embedding_exp), dim=1).view(len(binary_clauses_idx), 4 * global_embedding.shape[-1])
        x.append(self.common_layer(self.ternary_op_layer(binary_rep)).view(-1))

        # logging.info(f"local embedding: {local_embedding}")
        
        x = torch.cat(x)
        x = x.view(1, x.shape[-1])

        # x2 = local_embedding[]
        # x = F.softmax(x)

        return x

class NodeDecoder(nn.Module):

    def __init__(self, hidden_dim=cmd_args.hidden_dim):
        super().__init__()
        self.attr_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
        ) 

        self.var1_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
        )

        self.var2_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
        )

        self.rela_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
        )

        self.attr_or_rela_score_layer = nn.Sequential(
            nn.Linear(cmd_args.hidden_dim * 2, int(cmd_args.hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim), int(cmd_args.hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(cmd_args.hidden_dim/2), 1),
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
    
    def get_values(self, node_embeddings, graph, locs, layer):
        # graph_embedding is of #node * hidden dim

        node_num = node_embeddings.shape[0]
        locs = self.locs_to_byte(locs, node_num)

        embeddings = node_embeddings[locs]
        if (embeddings.shape[0] == 0):
            print("here!")
        values = layer(embeddings)
        
        return values, embeddings
 

    def is_attr(self, graph, name):
        return name in graph.attrs

    def get_attr_operation(self, name, configure):
        for key, value in configure["choices"].items():
            if name in value:
                return key 
        return 

    # def get_clause_idx(self, env, clause):
    #     clauses = env.actions

    #     for ct, cl in enumerate(clauses):
    #         if len(cl) == len(clause):
    #             for e1, e2 in cl, clauses:
    #                 if type(e1) == int and not type(e2) == int:
    #                     e1 = f"var_{e1}"
    #                 elif not type(e1) == int and  type(e2) == int:
    #                     e2 = f"var_{e2}"

    #                 if not (e1 == e2):
    #                     continue
                    
    #                 return ct 


    def forward(self, graph_embedding, state):

        graph = state.graph
        values = [None] * len(state.actions)

        local_embeddings = graph_embedding[0]
        global_embedding = graph_embedding[1]

        self.state = global_embedding
        node_embeddings = torch.stack([torch.cat((local_embedding, self.state.squeeze(0))) for local_embedding in local_embeddings])

        var1_locs = self.get_var_locs(graph)
        # var1_names = graph.vars.keys()
        var1_probs, var1_embeddings = self.get_values(node_embeddings, graph, var1_locs, self.var1_score_layer)
        var1_states = [ self.gru_cell(var1_embedding.reshape(1, -1)[:, :cmd_args.hidden_dim], self.state) for var1_embedding in var1_embeddings ]

        for var1_id, (var1_prob, current_state) in enumerate(zip(var1_probs, var1_states)):

            node_embeddings = torch.stack([torch.cat((local_embedding, current_state.squeeze(0))) for local_embedding in local_embeddings])
            var1_name = self.idx_to_var(graph, var1_id)
            attr_or_rela_locs = graph.get_attr_or_rela_locs()
            attr_or_rela_probs, attr_or_rela_embeddings = self.get_values(node_embeddings, graph, attr_or_rela_locs, self.attr_or_rela_score_layer)
            
            prob_or_rela_states = [self.gru_cell(attr_or_rela_embedding.reshape(1, -1)[:, :cmd_args.hidden_dim], self.state) for attr_or_rela_embedding in attr_or_rela_embeddings]
            
            for attr_or_rela_id, (attr_or_rela_prob, prob_or_rela_state) in enumerate(zip(attr_or_rela_probs, prob_or_rela_states)):
                node_embeddings = torch.stack([torch.cat((local_embedding, prob_or_rela_state.squeeze(0))) for local_embedding in local_embeddings])
                sel_rela_or_attr = graph.get_attr_or_rela_by_idx(attr_or_rela_id) 

                if self.is_attr(graph, sel_rela_or_attr):
                    operation = self.get_attr_operation(sel_rela_or_attr, graph.config)
                    clause_idx = state.action_dict[str([operation, sel_rela_or_attr, var1_name])]
                    values[clause_idx] = var1_prob + attr_or_rela_prob
                else:
                    var2_locs = graph.get_var_locs()
                    var2_locs.remove(var1_locs[var1_id])
                    var2_probs, _ = self.get_values(node_embeddings, graph, var2_locs, self.var2_score_layer)
                    nodes = graph.get_nodes()
                    var2_names = [int(nodes[v][-1]) for v in var2_locs]

                    for var2_name, var2_prob in zip(var2_names, var2_probs):
                        clause_idx = state.action_dict[str([sel_rela_or_attr, var1_name, var2_name])]
                        # decay is omited here
                        values[clause_idx] = var1_prob + attr_or_rela_prob + var2_prob

        return torch.stack(values).view(1, -1)


# decode a clause we get, or use other method instead
class GlobalDecoder(nn.Module):

    def __init__(self, encoder, embedding_layer, hidden_dim=cmd_args.hidden_dim):
        super().__init__()

        self.gru_cell = nn.GRUCell(hidden_dim, hidden_dim)
        self.encoder = encoder
        self.embedding_layer = embedding_layer
        
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

        

    def forward(self, graph_embeddings, state):
        
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

        graph = state.graph
        clause_embeddings = []

        for clause in state.actions:

            if clause[0] == "behind" or clause[0] == "right":
                embeddings = torch.tensor((self.encoder.get_embedding(clause[0]), 
                              self.encoder.get_embedding(f"var_{clause[1]}"), 
                              self.encoder.get_embedding(f"var_{clause[2]}")))
                embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[-1])
                embeddings = self.embedding_layer(embeddings)

                rela_embedding = embeddings[0]
                v1_embedding =   embeddings[1]
                v2_embedding =   embeddings[2]

                s1 = self.gru_cell(rela_embedding, self.state)
                s2 = self.gru_cell(v1_embedding, s1)
                s3 = self.gru_cell(v2_embedding, s2)
                clause_embeddings.append(s3)
                
            else:
                embeddings = torch.tensor((self.encoder.get_embedding(clause[1]), 
                                           self.encoder.get_embedding(f"var_{clause[2]}")))
                embeddings = embeddings.view(embeddings.shape[0], embeddings.shape[-1])
                embeddings = self.embedding_layer(embeddings)

                attr_embedding = embeddings[0]
                var_embedding =   embeddings[1]
                
                s1 = self.gru_cell(attr_embedding, self.state)
                s2 = self.gru_cell(var_embedding, s1)

                clause_embeddings.append(s2)

        clause_embeddings = torch.cat(clause_embeddings)
        values = self.value_layer(clause_embeddings).view(1, -1)

        return values

# Take in a batch of  local+global embedding of the nodes, then
# output the result for deciding the action
class DQPolicy(nn.Module):

    def __init__(self, dataset, decoder, hidden_dim=cmd_args.hidden_dim):
        super().__init__()
        embedding_layer = nn.Embedding(len(dataset.attr_encoder.lookup_list), cmd_args.hidden_dim)
        self.gnn = gnn_models[cmd_args.gnn_version](dataset, embedding_layer)
        if decoder == "ClauseDecoder":
            self.decoder = ClauseDecoder()
        elif decoder == "GlobalDecoder":
            self.decoder = GlobalDecoder(dataset.attr_encoder, self.gnn.embedding_layer)
        self.hidden_dim = hidden_dim

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, state):
        
        # local_embedding: tensor(# of nodes, hidden dim)
        # global_embedding: tensor(1, hidden dim)
        # clauses_idx: [[1,2],[3,4], [5,6,7], [8,9,0]]
        data = state.data
        graph_embeddings = self.gnn(data)
        action_values = self.decoder(graph_embeddings, state)
        return action_values


steps_done = 0
EPS_END = 0.01
EPS_DECAY = 1000

def select_action(policy_net, state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (cmd_args.eps - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    logging.info(f"eps_threshold: {eps_threshold}")
    clauses_idx = state.actions 
    avoid_idx = state.idx_selected

    selections = []
    # aviod repeat selections
    for idx in range(len(clauses_idx)):
        if idx not in avoid_idx:
            selections.append(idx)

    action_values = policy_net(state)
    # logging.info(f"all probs: {action_values}")

    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            logging.info("max")
            
            action_idx = action_values[:,selections].max(1)[1]
            action_idx = torch.tensor(selections[action_idx]).view(1, 1)
            action_value = action_values[:,selections].max(1)[0]
            logging.info(f"clause prob: {action_value}")
            return action_idx
    else:
        logging.info("random")
        select_id = random.choice(selections)
        action_value = action_values[0][select_id]
        logging.info(f"clause prob: {action_value}")
        return torch.tensor([[select_id]], dtype=torch.long)
