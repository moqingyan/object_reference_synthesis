import argparse
import sys
import random
import numpy as np
import torch
import os
import logging

class SampleDistr(object):

    def __init__(self):
        data_dir = os.path.abspath(__file__ + "../../../../data")
        default_model_dir = os.path.abspath(os.path.join(data_dir, f"./model"))

        self.parser = argparse.ArgumentParser(description='Argparser', allow_abbrev=True)
        self.parser.add_argument("--phase", default="training", type=str, help="phase is training / testing")
        self.parser.add_argument("--cuda_id", default=0, type=int, help="cuda id we want to use")

        self.parser.add_argument("--embedding_dim", default=64, type=int)
        self.parser.add_argument("--hidden_dim", default=64, type=int)
        self.parser.add_argument("--max_var_num", default=3, type=int)
        self.parser.add_argument("--max_obj_num", default=20, type=int)
        self.parser.add_argument("--log_name", default=None, type=str)
        self.parser.add_argument("--log_prefix", default="", type=str)
        self.parser.add_argument("--save_num", default=2, type=int, help="for every save_num of iteration processed, we save the model once")
        
        self.parser.add_argument("--graph_file_name", default="prob_unit_test_2.pkl", type=str)
        self.parser.add_argument("--dataset_name", default="prob_unit_test_2.pt", type=str)
        self.parser.add_argument("--scene_file_name", default="prob_unit_test_2.json")
        self.parser.add_argument("--test_set", default="prob_unit_test_2", type=str, help="preset test dataset to make life easier: prob_unit_test_2")

        self.parser.add_argument("--sub_loss", default="yes", type=str, help="whether use sub problems to calculate loss, yes/no")
        self.parser.add_argument("--max_sub_prob", default=20, type=int, help="The max number of sub problems to calculate loss, yes/no")
        
        self.parser.add_argument("--lr", default=0.00005, type=float)
        self.parser.add_argument("--eps", default=0.95, type=float)
        self.parser.add_argument("--eps_decay", default=0.1, type=float)
        self.parser.add_argument("--batch_size", default=5, type=int)
        self.parser.add_argument("--reward_penalty", default="yes", type=str, help='whether we are adding a penalty when the problem is not solved, yes/no')
        self.parser.add_argument("--reward_type", default="eliminated", type=str, help="reward type: 'preserved', 'eliminated', 'no_intermediate', 'only_success'")
        self.parser.add_argument("--var_space_constraint", default="no", type=str)
        self.parser.add_argument("--gamma", default=0.95, type=float, help="reward discount factor")


        self.parser.add_argument('--episode_iter', default=2, type=int, help='the total number of rollouts')
        self.parser.add_argument('--episode_length', default=10, type=int, help='the max rollout length in the rl process')
        self.parser.add_argument('--decay', default=0.95, type=float, help="the learning rates for the optimizers")
        self.parser.add_argument('--normalize_reward', default="no", type=str, help="whether to normalize the reward or not, yes/no")
        self.parser.add_argument('--model_dir', default=default_model_dir, type=str)

        self.parser.add_argument('--test_iter', default=100, type=int, help="number of test iterations taken")
        self.parser.add_argument('--test_type', default="sample", type=str, help="sample/max")
        self.parser.add_argument("--beam_size", default=30, type=int)
        self.parser.add_argument("--hard_constraint", default="yes", type=str)
        self.parser.add_argument("--seed", default=0, type=int)

        self.parser.add_argument("--prob_dataset", default="yes", type=str)
        self.parser.add_argument("--category_offsets", type=float, nargs='+', default=[0.5, 0.9, 1.0], help="categories offset")
        
        self.parser.add_argument("--global_node", default="no", type=str)
        self.parser.add_argument("--gnn_version", default="GNNGL", type=str)
        self.parser.add_argument("--max_node_num", default=100)
        self.parser.add_argument("--decoder_version", default="GlobalDecoder", type=str)

        self.parser.add_argument("--binding_node", default="no", type=str)
        self.parser.add_argument("--target_update", default=1000, type=int)

        self.args = self.parser.parse_args(sys.argv[1:])

def convert_str_to_bool(cmd_args):
    for key, val in vars(cmd_args).items():
        if val == "yes":
            setattr(cmd_args, key, True)
        elif val == "no":
            setattr(cmd_args, key, False)


sp = SampleDistr()
cmd_args = sp.args

cmd_args.model_save_dir = os.path.abspath(os.path.join(cmd_args.model_dir,
    f"./model-{cmd_args.log_prefix}-update{cmd_args.target_update}-{cmd_args.lr}-pen{cmd_args.reward_penalty}-{cmd_args.reward_type}-nor{cmd_args.normalize_reward}-{cmd_args.test_set}_{cmd_args.gnn_version}_{cmd_args.decoder_version}"))
if not os.path.exists(cmd_args.model_save_dir):
    os.mkdir(cmd_args.model_save_dir)

cmd_args.model_path = os.path.abspath(os.path.join(cmd_args.model_dir,
    f"./model-{cmd_args.log_prefix}-update{cmd_args.target_update}-{cmd_args.lr}-pen{cmd_args.reward_penalty}-{cmd_args.reward_type}-nor{cmd_args.normalize_reward}-{cmd_args.test_set}_{cmd_args.gnn_version}_{cmd_args.decoder_version}.pkl"))
    
data_dir = os.path.abspath(__file__ + "../../../../data")
log_dir = os.path.abspath(os.path.join(data_dir, f"./log"))
cmd_args.max_node_num = cmd_args.max_var_num + cmd_args.max_obj_num + cmd_args.max_obj_num * cmd_args.max_obj_num * 2 + 2 + cmd_args.max_var_num * cmd_args.max_obj_num + 15 * cmd_args.max_obj_num + 10

if not (type(cmd_args.test_set) == type(None)):
    cmd_args.graph_file_name = f"{cmd_args.test_set}_{cmd_args.lr}_{cmd_args.reward_type}_{cmd_args.gnn_version}_{cmd_args.decoder_version}.pkl"
    cmd_args.dataset_name = f"{cmd_args.test_set}_{cmd_args.lr}_{cmd_args.reward_type}_{cmd_args.gnn_version}_{cmd_args.decoder_version}.pt"
    cmd_args.scene_file_name = f"{cmd_args.test_set}.json"
    if cmd_args.log_name == None:
        cmd_args.log_name = f"{cmd_args.test_set}_lr_{cmd_args.lr}_{cmd_args.reward_type}_{cmd_args.gnn_version}_{cmd_args.decoder_version}.log"

print(cmd_args.target_update)
print(cmd_args.target_update == None)
print(cmd_args.log_name == None)

if cmd_args.log_name == None:
    if not cmd_args.target_update == None:
        cmd_args.log_path = os.path.abspath(os.path.join(log_dir, f"./{cmd_args.log_prefix}_update_{cmd_args.target_update}_{cmd_args.lr}_pen{cmd_args.reward_penalty}_pre{cmd_args.reward_type}_nor{cmd_args.normalize_reward}-{cmd_args.test_set}_{cmd_args.gnn_version}_{cmd_args.decoder_version}.log"))
    else:
        cmd_args.log_path = os.path.abspath(os.path.join(log_dir, f"./{cmd_args.log_prefix}_{cmd_args.lr}_pen{cmd_args.reward_penalty}_pre{cmd_args.reward_type}_nor{cmd_args.normalize_reward}_{cmd_args.test_set}_{cmd_args.gnn_version}_{cmd_args.decoder_version}.log"))
else:
    if not cmd_args.target_update == None:
        print("has target update")
        cmd_args.log_name = cmd_args.log_prefix + f"_update_{cmd_args.target_update}" + cmd_args.log_name
    else:
        cmd_args.log_name = cmd_args.log_prefix + cmd_args.log_names
    
    cmd_args.log_path = os.path.abspath(os.path.join(log_dir, cmd_args.log_name))

if cmd_args.prob_dataset == None:
    if "prob" in cmd_args.scene_file_name:
        cmd_args.prob_dataset = True
    else:
        cmd_args.prob_dataset = False

json_path = os.path.abspath(os.path.join(data_dir, f"./temp/query-{cmd_args.lr}-pen{cmd_args.reward_penalty}-pre{cmd_args.reward_type}-nor{cmd_args.normalize_reward}-{cmd_args.test_set}_{cmd_args.gnn_version}_{cmd_args.decoder_version}.json"))
cmd_args.json_path = json_path
convert_str_to_bool(cmd_args)

cmd_args.device = torch.device('cpu')
if torch.cuda.is_available():
    print(f"Setting cuda device to {cmd_args.cuda_id}")
    cmd_args.device = torch.device(f'cuda: {cmd_args.cuda_id}')
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    torch.cuda.set_device(cmd_args.cuda_id)

print(f"log path :{cmd_args.log_path}")
print(cmd_args)

if (sp.args.seed != None):
    random.seed(sp.args.seed)
    np.random.seed(sp.args.seed)
    torch.manual_seed(sp.args.seed)

logging.basicConfig(filename=cmd_args.log_path, filemode='w', level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logging.info(cmd_args)
logging.info("start!")
