import torch 
import torch.nn as nn 
import os 
import sys
import json
import pickle
from collections import namedtuple
import torch.nn.functional as F

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from scene2graph import Graph, GraphNode
from embedding import GNN, SceneDataset, GNNLocal, GNNGL, GNNGlobal
from cmd_args import cmd_args, logging
from decoder import Transition, ReplayMemory, DQPolicy, select_action, NodeDecoder, ClauseDecoder, GlobalDecoder
import decoder as DC
from utils import NodeType, EdgeType, Encoder, get_config, get_all_clauses, get_final_reward
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm
from env import Env
import copy 

class Learning_Model:

    def __init__(self, decoder, policy, memory, optimizer, current_it, steps_done):
        self.decoder = decoder
        self.policy = policy
        self.memory = memory
        self.optimizer = optimizer
        self.current_it = current_it
        self.steps_done = steps_done

steps = 0
# deep Q learning 
def optimize_model_DQ(memory, policy_net, target_net, optimizer):

    batch_size = cmd_args.batch_size
    gamma = cmd_args.gamma
    if len(memory) < cmd_args.batch_size:
        return

    transitions = memory.sample(batch_size)
    # logging.info(f"Transitions: {transitions}")
    
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

    global steps
    steps += 1
    
    if not type(cmd_args.max_gradient_steps) == type(None):
                    
        if steps % cmd_args.record_gradient_steps:
            logging.info(f"gradient step: {steps}")

        if steps >= cmd_args.max_gradient_steps:
            print("hit max gradient steps")
            logging.info("hit max gradient steps")
            stop = True
            raise SystemExit()

    # Optimize the model
    optimizer.zero_grad()

    loss.backward()

    for param in policy_net.named_parameters():
        if not type(param[1].grad) == type(None):
            param[1].grad.data.clamp_(-1, 1)
            # logging.info(f"name: {param[0]}, grad:{param[1].grad.data}]")

    
    optimizer.step()

def episode(policy, target, data_point, graph, config, attr_encoder, memory, total_count, optimizer):
    env = Env(data_point, graph, config, attr_encoder)
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
        return True
    else:
        return False

def train(dataset, graphs, config):

    if os.path.exists(cmd_args.model_path) and os.path.getsize(cmd_args.model_path) > 0:
        model = torch.load(cmd_args.model_path)
        # memory = model.memory
        # decoder = model.decoder
        # policy = model.policy
        # optimizer = model.optimizer
        # current_it = model.current_it
        DC.steps_done = model.steps_done
        
    else: 
        
        # decoder = ClauseDecoder()
        # decoder = GlobalDecoder(dataset.attr_encoder, )
        # decoder_name = "ClauseDecoder"
        decoder_name = "GlobalDecoder"
        policy = DQPolicy(dataset, decoder_name)
        decoder = policy.decoder
        current_it = 0
        memory = ReplayMemory(10000)
        optimizer = optim.RMSprop(policy.parameters(), lr=cmd_args.lr)
        DC.steps_done = 0
        model = Learning_Model(decoder, policy, memory, optimizer, current_it, DC.steps_done)

    # target_decoder = type(model.decoder)()
    decoder_name = str(type(model.policy.decoder).__name__)
    target = DQPolicy(dataset, decoder_name)
    target_decoder = target.decoder
    target.load_state_dict(model.policy.state_dict())
    target.eval()

    data_loader = DataLoader(dataset)

    for it in range(cmd_args.episode_iter):
        logging.info(f"training iteration: {it}")
        success_ct = 0
        total_loss = 0.0
        total_ct = 0

        if model.current_it > it:
            continue

        for data_point, ct in zip(data_loader, tqdm(range(len(data_loader)))):
            
            logging.info(f"task ct: {ct}")
            
            graph = graphs[data_point.graph_id]
            suc = episode(model.policy, target, data_point, graph, config, dataset.attr_encoder, model.memory, total_ct, model.optimizer)

            total_ct += 1
            if suc:
                success_ct += 1

        logging.info(f"success count: {success_ct}")

        # if it % cmd_args.save_num == 0:
        #     model.steps_done = DC.steps_done
        #     model.current_it = it
        #     torch.save(model, cmd_args.model_path)
            
        if it % cmd_args.save_num == 0:
            model.steps_done = DC.steps_done
            # model.eps = eps
            model.current_it = it
            model_name = f"model_{it}.pkl"
            model_path = os.path.join(cmd_args.model_save_dir, model_name)
            torch.save(model, model_path)

    print('Complete')
