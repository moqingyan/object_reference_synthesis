import torch 
import torch.nn as nn 
import os 
import sys
import json
import pickle
from collections import namedtuple
import torch.nn.functional as F
import torch.multiprocessing as mp
from functools import partial
import math

torch.multiprocessing.set_sharing_strategy('file_system')
common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from scene2graph import Graph, GraphNode
from embedding import GNN, SceneDataset, GNNLocal, GNNGL, GNNGlobal, create_dataset
from cmd_args import cmd_args, logging
from decoder import Transition, ReplayMemory, DQPolicy, select_action, NodeDecoder, ClauseDecoder
import decoder as DC
from utils import NodeType, EdgeType, Encoder, get_config, get_all_clauses, get_final_reward
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from env import Env
import copy 
import time
from query import SceneInterp

from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# deep Q learning 
def optimize_model_DQ(memory, policy_net, target_net, optimizer):

    batch_size = cmd_args.batch_size
    gamma = cmd_args.gamma
    if len(memory) < cmd_args.batch_size:
        return

    transitions = memory.sample(batch_size)
    
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
   
    # state_batch = batch.state
    # action_batch = torch.stack(batch.action).view(-1)
    # reward_batch = torch.stack(batch.reward).view(-1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = []
    for state, action, reward in zip(batch.state, batch.action, batch.reward):
        state_action_values.append(policy_net(state).gather(1, action)) 
    state_action_values = torch.cat(state_action_values)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.

    next_state_values = []
    for state in batch.next_state:

        if type(state) == type(None):
            next_state_values.append(torch.tensor([0.0]))
        else:
            unary_clauses_idx, binary_clauses_idx = state.get_clauses_idx()
            clauses_idx = unary_clauses_idx + binary_clauses_idx 
            next_idx_selected = state.idx_selected

            selections = []
            for idx in range(len(clauses_idx)):
                if idx not in next_idx_selected:
                    selections.append(idx)

            next_state_values.append(target_net(state)[:, selections].max(1)[0].detach())
    next_state_values = torch.cat(next_state_values)
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + torch.tensor(batch.reward)

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.view(-1, 1))
    logging.info(f"loss: {loss}")

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.named_parameters():
        if not type(param[1].grad) == type(None):
            param[1].grad.data.clamp_(-1, 1)
            # logging.info(f"name: {param[0]}, grad:{param[1].grad.data}]")
    optimizer.step()

def episode(policy, target, data_point, graph, config, attr_encoder, memory, total_count, optimizer):
    env = Env(data_point, graph, config, attr_encoder, ref=True, is_uncertain=True)
    
    for iter_count in range(cmd_args.episode_length):

        state = env.get_state()
        action = select_action(policy, state)

        logging.info(f"selected clause: {env.actions[action]}")
        next_state, reward, done, _ = env.step(action)

        if done: 
            next_state = None

        # cannot find out the result in limited steps
        if (iter_count == cmd_args.episode_length - 1):
            reward = get_final_reward(env)
        
        logging.info (f"reward: {reward}")
        memory.push(state, action, next_state, reward)
        optimize_model_DQ(memory, policy, target, optimizer)

        if done:
            break #
        
        if total_count % cmd_args.target_update == 0:
                target.load_state_dict(policy.state_dict())

    if env.success:
        return True, env.clauses, env.obj_poss_left[-1]
    else:
        return False, env.clauses[:-1], env.obj_poss_left[-2]

# for each of the problem, we start to overfit the exact problem
def cont_single(file_name, datapoint, model, graphs, save_dir, attr_encoder, config, save=True, n=10):

    file_path = os.path.join(save_dir, str(file_name))
    exist = os.path.exists(file_path)
    logging.info(f"task: {file_name}, {exist}")
    if exist:
        logging.info("skip")
        return
    
    total_ct = 0
    memory = ReplayMemory(10000)
    method = None
    
    policy = copy.deepcopy(model.policy)
    target = copy.deepcopy(policy)
    target.eval()

    optimizer = optim.RMSprop(policy.parameters(), lr=cmd_args.lr)

    success_progs = []
    all_progs = []

    eps = cmd_args.eps
    graph = graphs[datapoint.graph_id]

    for it in range(cmd_args.test_iter):
        
        suc, prog, obj_left = episode(policy, target, datapoint, graph, config, attr_encoder, memory, total_ct, optimizer)
        total_ct += 1

        if suc:
            success_progs.append(prog)
            method = "model"
            break

        if not suc:
            all_progs.append((prog, obj_left))

    logging.info(f"success: {success_progs}")

    # if not success, explot the best selections.
    scene_interpreter = SceneInterp(graph.ground_truth_scene, config)

    if len(success_progs) == 0:
        logging.info("start exploit")

        # take second element for sort
        candidates = sorted(all_progs, key=lambda x: x[1])
        for prog, obj_left in candidates:
            # no progress
            if len(prog) == 0:
                logging.info("empty prog to exploit, stop")
                break

            logging.info(f"prog: {prog}")
            logging.info(f"obj left: {obj_left}")
            prog_left = scene_interpreter.interp_enum(prog, datapoint.y, max_depth=2)

            if not type(prog_left) == type(None):
                print(prog_left)
                prog.append(prog_left)
                success_progs.append(prog)
                method = f"exploit: {len(prog_left)}"
                logging.info(f"We found: {success_progs} at depth {len(prog_left)}")
                break

    if save:
        file_path = os.path.join(save_dir, str(file_name))
        with open(file_path, 'w') as suc_prog_file:
            res = {}
            res['prog'] = success_progs
            res['method'] = method
            json.dump(res, suc_prog_file)

    return success_progs

# This is for multiprocessing.
def cont_multiple(model, dataset, graphs, save_dir, attr_encoder, config):
    num_process = 10
    pool = mp.Pool(num_process)
    dataloader = DataLoader(dataset)
    pool.starmap( partial(cont_single, model = model, graphs=graphs, save_dir=save_dir, attr_encoder=attr_encoder, config=config), enumerate(dataloader))

if __name__ == '__main__':

    DC.steps_done = 10000
    
    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))
    model_dir = os.path.abspath(os.path.join(data_dir, "model"))
    cont_res_dir = os.path.abspath(os.path.join(data_dir, "eval_result/DQN_prob_CLEVR_testing_1000"))

    scene_file_name = "img_test_prob_CLEVR_testing_data_val_1000.json"
    graph_file_name = "img_test_prob_CLEVR_testing_data_val_1000.pkl"
    dataset_name = "img_test_prob_CLEVR_testing_data_val_1000.pt"


    cmd_args.graph_file_name = graph_file_name
    cmd_args.scene_file_name = scene_file_name
    cmd_args.dataset_name = dataset_name
    cmd_args.test_iter = 200
    print(cmd_args)

    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, scene_file_name))
    graphs_path = os.path.join(raw_path, graph_file_name)

    # update the cmd_args corresponding to the info we have
    cmd_args.graph_file_name = graph_file_name

    lr4_10_model_path = os.path.join(model_dir, "model--update1000-0.0001-penyes-eliminated-norno-img_test_prob_CLEVR_training_data_val_1000_GNNGL_GlobalDecoder/model_80.pkl")
    model = torch.load(lr4_10_model_path, map_location=torch.device('cuda'))
    # config = get_config()
    # refrl = RefRL(scene_dataset, config, graphs)

    graphs, scene_dataset = create_dataset(data_dir, scenes_path, graphs_path)
    logging.info ("start cont training")

    dataloader = DataLoader(scene_dataset)
    for ct, datapoint in enumerate(dataloader):
        cont_single(ct, datapoint, model, graphs, cont_res_dir, scene_dataset.attr_encoder, scene_dataset.config, save=True, n=10)
    start_time = time.time()
    # cont_multiple(model, scene_dataset, graphs, cont_res_dir, scene_dataset.attr_encoder, scene_dataset.config)

    # for ct, datapoint in enumerate(dataloader):
    #     logging.info (ct)
    #     cont_single(ct, datapoint, model, graphs, cont_res_dir, scene_dataset.attr_encoder, scene_dataset.config)
    end_time = time.time()

    logging.info (f"finished_training in {end_time - start_time}")
