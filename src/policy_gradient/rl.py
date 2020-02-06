import os
import sys
import json
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time 
from copy import deepcopy

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from cmd_args import cmd_args, logging
from scene2graph import Graph2, GraphNode, Edge
from embedding import GNN, SceneDataset, GNNLocal, GNNGL, GNNGlobal
from torch_geometric.data import Data, DataLoader
from torch.autograd import Variable
from torch import autograd
from torch.distributions.categorical import Categorical
from tqdm import tqdm
from query import query
from utils import AVAILABLE_OBJ_DICT, policy_gradient_loss, get_reward, get_final_reward,  NodeType, EdgeType, Encoder, OneHotEmbedding, get_all_clauses
from env import Env 
from decoder import NodeDecoder, GlobalDecoder, AttDecoder
from itertools import chain

gnn_models = {}
gnn_models["GNNLocal"] = GNNLocal
gnn_models["GNNGL"] = GNNGL
gnn_models["GNNGlobal"] = GNNGlobal

class GDPolicy(nn.Module):

    def __init__(self, gnn, encoder, eps=cmd_args.eps, hidden_dim = cmd_args.hidden_dim):
        super().__init__()
        self.gnn = gnn
        self.encoder = encoder
        # exchangable get_obj_layer for decoder
        # self.decoder = Decoder(self.encoder, gnn.embedding_layer)
        self.hidden_dim = hidden_dim
        self.reward_history = []
        self.prob_history = Variable(torch.Tensor())
        self.eps = eps
        self.decoder = None

    def get_prob_clause(self, graph_embedding, env, phase):
        raise NotImplementedError
        
    def reset(self, eps=cmd_args.eps):
        self.eps = eps
        self.reward_history = []
        self.prob_history = Variable(torch.Tensor())

    def forward(self, env, phase = "train"):
        # graph_embedding = self.gnn(env.data)
        # prob, clause = self.decoder(graph_embedding, self.encoder, self.eps, phase)
        prob, clause = self.get_prob_clause(env, phase)
        
        if self.prob_history.dim() != 0:
            self.prob_history = torch.cat([self.prob_history, prob])
        else:
            self.prob_history = (prob)

        env.clauses.append(clause)
        reward = env.step()
        if env.is_finished() and cmd_args.reward_type == "only_success":
            if reward == -1:
                self.reward_history = [0] * len(self.reward_history)

        self.reward_history.append(reward)

        return env

class NodeSelPolicy(GDPolicy):

    def __init__(self, gnn, encoder, eps=cmd_args.eps, hidden_dim = cmd_args.hidden_dim):
        super().__init__(gnn, encoder, eps, hidden_dim)
        self.decoder = NodeDecoder()
    
    def get_prob_clause(self, env, phase = "train"):
        def get_refs(clauses):
            refs = [0]
            for clause in clauses:
                if 1 in clause and not 1 in refs:
                    refs.append(1)
                if 2 in clause and not 2 in refs:
                    refs.append(2)
            return refs
        ref = get_refs(env.clauses)
        graph_embedding = self.gnn(env.data)
        if type(self.decoder) == type(None):
            prob, clause = self.clause_decoder(graph_embedding, env.graph, ref, self.eps, phase)
        else:
            prob, clause = self.decoder(graph_embedding, env.graph, ref, self.eps, phase)

        # prob, clause = self.decoder(graph_embedding, env.graph, ref, self.eps, phase)
        return prob, clause

# class GlobalPolicy(GDPolicy):

#     def __init__(self, gnn, encoder, eps=cmd_args.eps, hidden_dim = cmd_args.hidden_dim):
#         super().__init__(gnn, encoder, eps, hidden_dim)
#         self.decoder = GlobalDecoder(self.encoder, gnn.embedding_layer)
    
#     def get_prob_clause(self, env, phase = "train"):
#         graph_embedding = self.gnn(env.data)
#         prob, clause = self.decoder(graph_embedding, env.graph, self.encoder, self.eps, phase)
#         return prob, clause

class GlobalPolicy(GDPolicy):

    def __init__(self, gnn, encoder, eps=cmd_args.eps, hidden_dim = cmd_args.hidden_dim):
        super().__init__(gnn, encoder, eps, hidden_dim)
        self.decoder = GlobalDecoder(self.encoder, gnn.embedding_layer)
    
    def get_prob_clause(self, env, phase = "train"):
        graph_embedding = self.gnn(env.data)
        prob, clause = self.decoder(graph_embedding, env.graph, self.eps)
        return prob, clause

class AttPolicy(GDPolicy):

    def __init__(self, gnn, encoder, eps=cmd_args.eps, hidden_dim = cmd_args.hidden_dim):
        super().__init__(gnn, encoder, eps, hidden_dim)
        self.decoder = AttDecoder(self.encoder, gnn.embedding_layer)
    
    def get_prob_clause(self, env, phase = "train"):
        graph_embedding = self.gnn(env.data)
        prob, clause = self.decoder(graph_embedding, self.eps, phase)
        return prob, clause


decoder_models = {}
decoder_models["NodeDecoder"] = NodeSelPolicy
decoder_models["GlobalDecoder"] = GlobalPolicy
decoder_models["AttDecoder"] = AttPolicy

class RefRL():

    def __init__(self, dataset, config, graphs, hidden_dim=cmd_args.hidden_dim):
        
        # embedding_layer = nn.Embedding(len(dataset.attr_encoder.lookup_list), cmd_args.hidden_dim)
        embedding_layer = OneHotEmbedding(len(dataset.attr_encoder.lookup_list), cmd_args.hidden_dim)
        
        self.gnn = gnn_models[cmd_args.gnn_version](dataset, embedding_layer)
        self.encoder = dataset.attr_encoder
        self.gradient_steps = 0

        self.graphs = graphs
        self.dataset = dataset
        train_num = int(len(dataset)*0.8)
        if train_num == len(dataset):
            train_num = -1
        # self.train_data = self.dataset[:train_num]
        # self.test_data = self.dataset[train_num:]
        self.train_data = self.dataset

        self.config = config
        self.hidden_dim = hidden_dim

        self.policy = decoder_models[cmd_args.decoder_version](self.gnn, self.encoder)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=cmd_args.lr)
        print(self.policy)
        self.episode_iter = cmd_args.episode_iter

        # for save and load
        self.iteration = 0

    def episode(self, data_point, graph, eps, attr_encoder, phase="train"):
        self.policy.reset(eps)
        retrain_list = []

        env = Env(data_point, graph, self.config, attr_encoder)
        iter_count = 0

        while not env.is_finished():
            # cannot figure out the clauses in limited step
            if (iter_count > cmd_args.episode_length):
                if cmd_args.reward_type == "only_success":
                    final_reward = get_final_reward(env)
                    if final_reward == -1:
                        self.policy.reward_history = [0.0] * len(self.policy.reward_history)
                        self.policy.reward_history.append(-1.0)
                    else:
                        self.policy.reward_history.append(1.0)
                    self.policy.reward_history = torch.tensor(self.policy.reward_history)
                else:
                    self.policy.reward_history.append(get_final_reward(env))
                    self.policy.prob_history = torch.cat([self.policy.prob_history, torch.tensor([1.0])])
                break

            iter_count += 1
            env = self.policy(env, phase)

            if cmd_args.sub_loss:
                
                if not env.is_finished():
                    retrain_prob =  (env.obj_poss_left[-2] - env.obj_poss_left[-1])/env.obj_poss_left[0]
                    decision = np.random.choice([0,1], 1, p=[1-retrain_prob, retrain_prob])
                    if decision[0] == 1:
                        retrain_list.append((deepcopy(env.data), deepcopy(env.clauses)))
            
        logging.info(self.policy.reward_history)
        logging.info(self.policy.prob_history)
        logging.info(env.clauses)
        return env, retrain_list

def fit_one(refrl, data_point, graph, eps):
    global sub_ct
    sub_ct += 1

    if sub_ct > cmd_args.max_sub_prob:
        return None, None 

    env, retrain_list = refrl.episode(data_point, graph, eps, refrl.dataset.attr_encoder)
    loss = policy_gradient_loss(refrl.policy.reward_history, refrl.policy.prob_history)
    
    if cmd_args.sub_loss:
        sub_loss = []
        for data_point, clauses in retrain_list:
            logging.info(f"clauses in env: {clauses}")
            e, r = fit_one(refrl, data_point, env.graph, eps)
            if e == None:
                print("Oops! running out of budget")
                return env, loss

            sub_loss.append(r)
        loss += sum(sub_loss)

    return env, loss

def fit(refrl):
    refrl.policy.train()
    # refrl.train_data.shuffle()
    total_ct = 0
    data_loader = DataLoader(refrl.train_data)
    eps = cmd_args.eps
    print(type(eps))

    # with autograd.detect_anomaly():
    for it in range(cmd_args.episode_iter):
        
        logging.info(f"training iteration: {it}")
        success = 0
        
        total_loss = 0.0 
        if refrl.iteration > it:
            continue

        for data_point, ct in zip(data_loader, tqdm(range(len(data_loader)))):
            global sub_ct 
            sub_ct = 0
            total_ct += 1
            logging.info(ct)
            
            # if ct == 20000:
            #     print ("debug")
            #     continue
            graph = refrl.graphs[data_point.graph_id]
            env, loss = fit_one(refrl, data_point, graph, eps)
            
            total_loss += loss
            if env.success:
                success += 1
            
            if total_ct % cmd_args.batch_size == 0:

                refrl.optimizer.zero_grad()
            
            loss.backward()
            
            if total_ct % cmd_args.batch_size == 0:
                refrl.optimizer.step()
                refrl.gradient_steps += 1
                
                if not type(cmd_args.max_gradient_steps) == type(None):
                    
                    if refrl.gradient_steps % cmd_args.record_gradient_steps:
                        logging.info(f"gradient step: {refrl.gradient_steps}")

                    if refrl.gradient_steps >= cmd_args.max_gradient_steps:
                        print("hit max gradient steps")
                        logging.info("hit max gradient steps")
                        stop = True
                        raise SystemExit()


            if total_ct % cmd_args.save_num == 0 and not total_ct == 0:
                torch.save(refrl, cmd_args.model_path)

        logging.info(f"at train iter {it}, success num {success}, ave loss {total_loss/ct}")
        refrl.iteration += 1
        eps = cmd_args.eps_decay * eps

def test(refrl, split="test"):
    
    logging.info(f"testing on {split} data")
    refrl.policy.eval()

    if split == "train":
        data_loader = DataLoader(refrl.train_data)
    else:
        data_loader = DataLoader(refrl.test_data)

    success = 0
    avg_loss = 0
    eps = 0
    total_ct = 0

    for it in range(cmd_args.test_iter):
        logging.info(f"testing iteration: {it}")
        with torch.no_grad():
            for data_point, ct in zip(data_loader, tqdm(range(len(data_loader)))):
                
                logging.info(ct)
                total_ct += 1

                graph = refrl.graphs[data_point.graph_id]
                env, _ = refrl.episode(data_point, graph, eps, refrl.dataset.attr_encoder, phase="test" )
                loss = policy_gradient_loss(refrl.policy.reward_history, refrl.policy.prob_history)
                avg_loss += loss

                if env.success:
                    success += 1

    avg_loss /= total_ct
    logging.info(f"Testing {split}: success {success} out of {total_ct}, average loss is {avg_loss}")

# if __name__ == "__main__":
#     data_dir = os.path.abspath(__file__ + "../../../data")
#     root = os.path.abspath(os.path.join(data_dir, "./processed_dataset"))

#     raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
#     graphs_path = os.path.join(raw_path, cmd_args.graph_file_name)
#     with open(graphs_path, 'rb') as graphs_file:
#         graphs = pickle.load(graphs_file)

#     config_path = os.path.join(data_dir, "config.json")
#     with open(config_path, 'r') as config_file:
#         config = json.load(config_file)
    
#     if cmd_args.max_var_num == 1:
#         for rela in ["left", "right", "front", "behind"]:
#             config["operation_list"].remove(rela)

#     scene_dataset = SceneDataset(root, config)
#     logging.info("Finished loading data")
    
#     # test_train(gnn, config, scenes[0])
#     # print("Finished unit test: train")

#     # refrl = RefRL(scene_dataset, config, graphs)
#     if os.path.exists(cmd_args.model_path) and os.path.getsize(cmd_args.model_path) > 0:
#         refrl = torch.load(cmd_args.model_path)
#         logging.info("Loaded refrl model")
#     else:
#         refrl = RefRL(scene_dataset, config, graphs)
#         logging.info("Constructed refrl model")

#     logging.info("Start testing")
#     start_time = time.time()
#     test(refrl)
#     test(refrl, "train")
#     end_time = time.time()
#     logging.info(f"Finished testing in {end_time - start_time}")
    
#     # logging.info("Start training")
#     # start_time = time.time()
#     # fit(refrl)
#     # end_time = time.time()
#     # logging.info(f"Finished training in {end_time - start_time}")

#     # logging.info("Start testing")
#     # start_time = time.time()
#     # test(refrl)
#     # test(refrl, "train")
#     # end_time = time.time()
#     # logging.info(f"Finished testing in {end_time - start_time}")

#     print ("Done")

