import os
import sys
import json 
import torch 
from torch_geometric.data import Data, DataLoader
import heapq
from dataclasses import dataclass, field
from typing import Any
import random 
import torch.multiprocessing as mp
import numpy as np
from functools import partial
import time 

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
src_path = os.path.abspath(os.path.join(__file__, "../../"))

print(common_path)
print(src_path)
sys.path.append(common_path)
sys.path.append(src_path)

from cmd_args import cmd_args, logging
from env import Env
from embedding import GNN, SceneDataset, create_dataset
from utils import get_config, AVAILABLE_OBJ_DICT, policy_gradient_loss, get_reward, get_final_reward,  NodeType, EdgeType, Encoder
from copy import deepcopy
from torch.multiprocessing import Pool
import torch.multiprocessing
# multiprocessing.set_start_method('spawn', True)

def episode(policy, data_point, graph, eps, attr_encoder, config, phase="train"):
    policy.reset(eps)
    retrain_list = []

    env = Env(data_point, graph, config, attr_encoder)
    iter_count = 0

    while not env.is_finished():
        # cannot figure out the clauses in limited step
        if (iter_count > cmd_args.episode_length):
            if cmd_args.reward_type == "only_success":
                final_reward = get_final_reward(env)
                if final_reward == -1:
                    policy.reward_history = [0.0] * len(policy.reward_history)
                    policy.reward_history.append(-1.0)
                else:
                    policy.reward_history.append(1.0)
                policy.reward_history = torch.tensor(policy.reward_history)
            else:
                policy.reward_history.append(get_final_reward(env))
                policy.prob_history = torch.cat([policy.prob_history, torch.tensor([1.0])])
            break

        iter_count += 1
        env = policy(env, phase)

        if cmd_args.sub_loss:
            
            if not env.is_finished():
                retrain_prob =  (env.obj_poss_left[-2] - env.obj_poss_left[-1])/env.obj_poss_left[0]
                retrain_prob = max(0.0, retrain_prob)
                decision = np.random.choice([0,1], 1, p=[1-retrain_prob, retrain_prob])
                if decision[0] == 1:
                    retrain_list.append((deepcopy(env.data), deepcopy(env.clauses)))
        
    logging.info(policy.reward_history)
    logging.info(policy.prob_history)
    logging.info(env.clauses)
    return env, retrain_list

def fit_one(policy, datapoint, graph, eps, attr_encoder, config):
    global sub_ct
    sub_ct += 1

    if sub_ct > cmd_args.max_sub_prob:
        return None, None 

    env, retrain_list = episode(policy, datapoint, graph, eps, attr_encoder, config)
    loss = policy_gradient_loss(policy.reward_history, policy.prob_history)
    
    if cmd_args.sub_loss:
        sub_loss = []
        for data_point, clauses in retrain_list:
            logging.info(f"clauses in env: {clauses}")
            e, r = fit_one(policy, datapoint, env.graph, eps, attr_encoder, config)
            if e == None:
                print("Oops! running out of budget")
                return env, loss

            sub_loss.append(r)
        loss += sum(sub_loss)

    return env, loss

# for each of the problem, we start to overfit the exact problem
def cont_single(file_name, datapoint, policy, graphs, save_dir, attr_encoder, config, save=True):

    policy = deepcopy(policy)
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=cmd_args.lr)

    success_progs = []
    global sub_ct 
    eps = cmd_args.eps
    graph = graphs[datapoint.graph_id]

    for it in range(cmd_args.episode_iter):
        sub_ct = 0
        env, loss = fit_one(policy, datapoint, graph, eps,  attr_encoder, config)
        if env.success:
            success_progs.append(env.clauses)
        
        if it % cmd_args.batch_size == 0:
            optimizer.zero_grad()

        loss.backward()

        if it % cmd_args.batch_size == 0:
            optimizer.step() 
    
        if len(success_progs) > 0:
            break

    if save:
        file_path = os.path.join(save_dir, str(file_name))
        with open(file_path, 'w') as suc_prog_file: 
            json.dump(success_progs, suc_prog_file)
    
    print(success_progs)
    return success_progs

# def meta_f(policy, graphs, save_dir):
#     def use_cont_single(datapoint, data_ct):
#         cont_single(policy, datapoint, graphs, )
#     return use_cont_single

def cont_multiple(policy, dataset, graphs, save_dir, attr_encoder, config):
    num_process = mp.cpu_count() - 1 
    pool = mp.Pool(num_process)
    dataloader = DataLoader(dataset)

    pool.starmap( partial(cont_single, policy = policy, graphs=graphs, save_dir=save_dir, attr_encoder=attr_encoder, config=config), enumerate(dataloader)) 

if __name__ == '__main__':

    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
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
    
    # update the cmd_args corresponding to the info we have 
    cmd_args.graph_file_name = graph_file_name

    lr4_10_model_path = os.path.join(model_dir, "refrl-0.0001-penyes-no_intermediate-norno-img_test_30.pkl")
    refrl = torch.load(lr4_10_model_path)

    print ("start training")
    
    dataloader = DataLoader(scene_dataset)
    # for ct, datapoint in enumerate(dataloader):
    #     cont_single(refrl.policy, datapoint, graphs, os.path.join(cont_res_dir, str(ct)), refrl.dataset.attr_encoder, refrl.config)
    start_time = time.time()
    cont_multiple(refrl.policy, scene_dataset, graphs, cont_res_dir, refrl.dataset.attr_encoder, refrl.config)
    # for ct, datapoint in enumerate(dataloader):
    #     cont_single(ct, datapoint, refrl.policy, graphs, cont_res_dir, refrl.dataset.attr_encoder, refrl.config)
    end_time = time.time()

    print (f"finished_training in {end_time - start_time}")