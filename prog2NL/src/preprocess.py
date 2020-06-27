import os
import json
import sys

common_path = os.path.abspath(os.path.join(os.path.abspath(__file__) + "../../../../src/common"))
print(common_path)
sys.path.append(common_path)

from query import SceneInterp
from utils import get_config

unary = ["shape", "size", "color", "material"]
def remove_one(scene_interp, prog):
    for clause in prog:
        new_prog = list(prog)
        new_prog.remove(clause)
        binding_dict = scene_interp.fast_query(new_prog)
        if (len(binding_dict["var_0"]) == 1):
            return new_prog
    return None

def shrink(prog, scene, config):
    scene_interp = SceneInterp(scene, config)
    last_prog = prog

    while (True):
        prog = remove_one(scene_interp, prog)

        # is minimal
        if type(prog) == type(None):
            return last_prog
        last_prog = prog

def remove_dup(prog):

    res = []
    for clause in prog:
        if clause not in res:
            res.append(clause)
    return res

def get_ref(prog):
    ref = set()
    for clause in prog:
        for e in clause:
            if type(e) == int:
                ref.add(e)
    return list(ref)

def add_attr(prog, scene):
    scene_interp = SceneInterp(scene, config)
    state = scene_interp.get_state(prog)

    bd_list = scene_interp.get_valid_binding(state)
    new_clauses = []
    ref = get_ref(prog)

    for var, obj in enumerate(bd_list):
        if var not in ref:
            continue
        color = scene["objects"][obj]["color"]
        shape = scene["objects"][obj]["shape"]
        new_clauses.append(["color", color, var])
        new_clauses.append(["shape", shape, var])

    prog += new_clauses
    return prog

def int_of_string(s):
    if type(s) == int:
        return s
    if s == "var_0" :
        return 0
    if s == "var_1" :
        return 1
    if s == "var_2" :
        return 2
    raise Exception(f"not valid string: {s}")

def flatten_prog (prog):

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

    if type(prog[-1][0]) == list:
        prog = prog[:-1] + prog[-1]

    new_prog = []
    for clause in prog:
        if clause[0] in unary:
            new_prog.append([clause[0], clause[1], int_of_string(clause[2])])
        else:
            new_prog.append([clause[0], int_of_string(clause[1]), int_of_string(clause[2])])

    return new_prog

def get_scenes_dict(scenes):
    scene_dict = {}
    for sct, scene in enumerate(scenes):
        scene_dict[scene["image_filename"]] = sct
    return scene_dict

if __name__ == "__main__":
    data_path = os.path.abspath(os.path.join(os.path.abspath(__file__) + "../../../../data"))
    from_dir = os.path.join(data_path, "eval_result/cont_res_DQN_False_Test_CLEVR_ref_train_questions_1000")
    to_dir = os.path.join(data_path, "eval_result/mini-clevr-ref-nl")
    image_dir = os.path.join(data_path, "mini-clevr-ref/image")
    scene_path = os.path.join(data_path, "mini-clevr-ref/scenes/CLEVR_ref_train_questions_2000.json")

    to_shrink_prog = False
    to_no_dup = True
    to_add_attr = True

    if not os.path.exists(to_dir):
        os.mkdir(to_dir)

    with open (scene_path, 'r') as scene_file:
        scenes = json.load(scene_file)
    config = get_config()
    scene_dict = get_scenes_dict(scenes)

    for f in os.listdir(from_dir):
        curr_file = os.path.join(from_dir, f)
        new_file = os.path.join(to_dir, f)

        with open (curr_file, 'r') as prog_file:
            prog_info = json.load(prog_file)
            if type(prog_info) == dict:
                new_prog = flatten_prog(prog_info["prog"])
            else:
                new_prog = flatten_prog(prog_info)
            print (new_prog)

        image_file = prog_info["image_name"]
        scene = scenes[scene_dict[image_file]]

        if to_shrink_prog:
            new_prog = shrink (new_prog, scene, config)
            print (f"shrinked: {new_prog}")

        if to_no_dup:
            new_prog = remove_dup(new_prog)
            print (f"no dup: {new_prog}")

        if to_add_attr:
            new_prog = add_attr(new_prog, scene)
            print (f"add attr: {new_prog}")

        with open (new_file, 'w') as new_prog_file:
            json.dump(new_prog, new_prog_file)