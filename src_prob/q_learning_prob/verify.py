import os 
import sys 
import json

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
sys.path.append(common_path)

from query import SceneInterp
from utils import get_config 

def check_success(binding_dict, target):
    if "var_0" not in binding_dict.keys():
        return False

    if not len(binding_dict["var_0"]) == 1:
        return False

    if list(binding_dict["var_0"])[0] == target:
        return True

    return False 

def check_prog_correct(scene, prog, target, config):
    interp = SceneInterp(scene["ground_truth"], config, is_uncertain=False)
    binding_dict = interp.fast_query(prog)
    res = check_success(binding_dict, target)
    return res

def analysis(scenes, progs_dir):
    ct = 0
    corrects = []
    wrongs = []
    config = get_config()

    for scene in scenes:
        target = 0
        for obj in scene["objects"]:
            
            if not ct == 6384:
                ct += 1
                target += 1
                continue

            prog_path = os.path.join(progs_dir, str(ct))
            prog_path_1 = os.path.join(progs_dir, str(ct)+"_1")

            if os.path.exists(prog_path_1):
                prog_path = prog_path_1

            with open(prog_path, 'r') as prog_file:
                prog_info = json.load(prog_file)
                prog = prog_info["prog"]

            print(ct)
            if type(prog[0][0]) == list:
                if len(prog) == 2:
                    prog[0].append(prog[1])
                elif len(prog) == 1:
                    if len(prog[0]) > 1 and type(prog[0][-1][0]) == list:
                        prog = prog[0][:-1] + (prog[0][-1])
                    else:
                        prog = prog[0]
                else:
                    raise("Prog formatting error")

            res = check_prog_correct(scene, prog, target, config)
            
            ct += 1
            target += 1

            if res:
                corrects.append(ct)
            else:
                print("wrong")
                wrongs.append(ct)

    # print(len(corrects))
    print(wrongs)

if __name__ == "__main__":

    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, "img_test_prob_CLEVR_testing_data_val_1000.json"))
    progs_dir =  os.path.abspath(os.path.join(data_dir, "eval_result/DQN_prob_CLEVR_testing_1000"))

    with open(scenes_path) as scenes_file:
        scenes = json.load(scenes_file)
    
    analysis(scenes, progs_dir)