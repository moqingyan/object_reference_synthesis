import os
import subprocess
# This is script as the entry point over different scripts in the whole folder

# --------------------- Ablation -----------------------
model = "LSTM" # option not available for prob dataset
# model = "policy"
# model = "DQN"

prob_dataset = False
# prob_dataset = True

# phase = "Train"
phase = "Test"

# You can put any of the dataset name under processed_dataset/raw here
# for either training or testing purpose
if prob_dataset:
    test_set = "prob_unit_test_2"
else:
    test_set = "unit_test_3"

# ---------------------- Learning params ------------------
# There are more learning parameters we can adjust, see src/common/cmd_args
# and src_prob/common/cmd_args 
# These are the most commonly used ones, and I have set the values to
# our common experiment setup

lr = 0.0001

# eps = 0.95 # the starting eps for training
eps = 0.1 # the staring eps for testing 
episode_iter = 20000 # The number of training rollouts
test_iter = 100 # the number of testing rollouts

eps_decay = 0.99 # policy gradient: control the eps decay rate
target_update = 1000 # DQN: When to update the target net

# ---------------------- Model Usage -------------------
current_dir = os.path.abspath(__file__)
model_dir = 'model/'
model_name = None
cont_res_name = None

model_dir_arg = None
model_name_arg = None
cont_res_name_arg = None

if type(cont_res_name) == type (None) and phase == "Test":
    cont_res_name = f"cont_res_{model}_{prob_dataset}_{phase}_{test_set}"
    cont_res_name_arg = f"--cont_res_name {cont_res_name}"

# set the default model dir according to the ablation setting 
if not type(model_name) == type(None):
    # set to costum model_dir and model_name
    model_dir_arg = f"--model_dir {model_dir}"
    model_name_arg = f"--model_name {model_name}"

else:
    # set the default model for evaluation
    if phase == "Test" :
        if model == "LSTM" and not prob_dataset:
            model_dir_arg = f"--model_dir LSTM_model"
            model_name_arg = "--model_name 3_1_1_1_1_LSTM.pkl"
            
        elif model == "DQN" and not prob_dataset:
            model_dir_arg = f"--model_dir DQN_model"
            model_name_arg = "--model_name 3_1_1_1_1_DQN.pkl"

        elif model == "DQN" and prob_dataset:
            model_dir_arg = f"--model_dir DQN_prob_model"
            model_name_arg = "--model_name CLEVR_prob.pkl"

# ---------------------- file to run -------------------------
if prob_dataset and model == "DQN" and phase == "Train":
    runner = "../src_prob/q_learning_prob/main.py"
elif prob_dataset and model == "DQN" and phase == "Test":
    runner = "../src_prob/q_learning_prob/cont_train.py"
# The policy_prob does not support continous training yet
elif prob_dataset and model == "policy" and phase == "Train": 
    runner = "../src_prob/policy_gradient/main.py"
elif not prob_dataset and model == "DQN" and phase == "Train":
    runner = "../src/q_learning/main.py"
elif not prob_dataset and model == "DQN" and phase == "Test":
    runner = "../src/q_learning/cont_train.py"
elif not prob_dataset and model == "LSTM" and phase == "Train":
    runner = "../src/LSTM/main.py"
elif not prob_dataset and model == "LSTM" and phase == "Test":
    runner = "../src/LSTM/cont_train.py"
elif not prob_dataset and model == "policy" and phase == "Train":
    runner = "../src/policy_gradient/main.py"
else:
    raise Exception("Wrong runner setting. Check whether the combination is valid")

runner_path = os.path.abspath(os.path.join(current_dir, runner))
print(runner_path)

# ---------------------- Set the parameters -------------------
test_set_arg = f"--test_set {test_set}"
lr_arg = f"--lr {lr}"
eps_arg = f"--eps {eps}"
episode_iter_arg = f"--episode_iter {episode_iter}"
test_iter_arg = f"--test_iter {test_iter}"
eps_decay_arg = f"--eps_decay {eps_decay}"
target_update_arg = f"--target_update {target_update}"
prob_dataset_arg = f"--prob_dataset {'yes' if prob_dataset else 'no'}"

# ------------------- Subprocess Call --------------------------
all_args = [test_set_arg, lr_arg, eps_arg, episode_iter_arg, test_iter_arg, eps_decay_arg, target_update_arg, prob_dataset_arg, model_dir_arg, model_name_arg, cont_res_name_arg]
used_args = [ arg.split(' ') for arg in list(filter(lambda arg:not type(arg) == type(None), all_args))] 

command = ["python", runner_path] + sum(used_args, []) 
print(command)
subprocess.call(command)