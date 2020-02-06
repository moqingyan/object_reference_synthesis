import json
import os
import sys
import random 
from functools import reduce 

common_path = os.path.abspath(os.path.join(__file__, "../../src/common"))
src_path = os.path.abspath(os.path.join(__file__, "../../src"))
sys.path.append(common_path)
sys.path.append(src_path)

from utils import get_config

align = False

class ThingsBank():

    def __init__(self, config):
        self.config = config
        self.setup()

    def setup(self):
        len_choice_list = []
        for attr, choices in self.config["choices"].items():
            len_choice_list.append(len(choices))
        self.len_choice_list = len_choice_list
        self.total_ct = reduce((lambda x, y: x * y), len_choice_list)

    def get_thing(self, obj_idx):

        def split_idx(cur_idx, len_choice_list):
            if len(len_choice_list) <= 1:
                return [cur_idx]
            possible_comb = reduce((lambda x, y: x * y), len_choice_list[1:])
            hd = int(cur_idx / possible_comb)
            new_idx = cur_idx - possible_comb * hd
            tl = split_idx(new_idx, len_choice_list[1:])
            return [hd] + tl

        if obj_idx > self.total_ct:
            return {}
        idxes = split_idx(obj_idx, self.len_choice_list)
        thing = {}
        for idx, (attr, choices) in enumerate(config["choices"].items()):
            thing[attr] = choices[idxes[idx]]
        return thing

    # unique_things_num == -1: no constraint on unique items
    # max_same_things_num == -1: no constraint on same items

    def get_things_by_uniq_list(self, unique_list):
        
        total_selection = list(range(self.total_ct))
        random.shuffle(total_selection)
        unique_things_idxes = total_selection[:len(unique_list)]
        
        things = {}
        for ct, idx in enumerate(unique_things_idxes):
            things[idx] = unique_list[ct]

        things_ls = []
        for idx, ct in things.items():
            things_ls += [idx] * ct
        random.shuffle(things_ls)

        things_detail_ls = [ self.get_thing(idx) for idx in things_ls]
        if not (len(things_detail_ls) == 7):
            print("here!!!!!")
            print(things)
            print(things_ls)
        return things_detail_ls

    def get_things(self, unique_things_num, max_same_things_num, total_things_num):
        
        if max_same_things_num < 1:
            max_same_things_num = 1
        if unique_things_num < 1:
            unique_things_num = total_things_num - max_same_things_num + 1
        if unique_things_num * max_same_things_num < total_things_num:
            raise ValueError
        
        things = {}
        total_selection = list(range(self.total_ct))
        random.shuffle(total_selection)
        unique_things_idxes = total_selection[:unique_things_num]

        for idx in unique_things_idxes:
            things[idx] = 1

        rest_choices = list(unique_things_idxes)
        for ct in range(total_things_num - unique_things_num):

            max_key = max(things, key=things.get)
            max_val = things[max_key]
            rest_pos = total_things_num - unique_things_num - ct 
            if (rest_pos) == (max_same_things_num - max_val):
                things[max_key] += rest_pos
                break 

            next_thing = random.choice(rest_choices)
            things[next_thing] += 1 
            if things[next_thing] == max_same_things_num:
                rest_choices.remove(next_thing)

        things_ls = []
        for idx, ct in things.items():
            things_ls += [idx] * ct
        random.shuffle(things_ls)

        things_detail_ls = [ self.get_thing(idx) for idx in things_ls]

        return things_detail_ls

def get_space_relations(things, align):
    x_pos = list(range(len(things)))
    random.shuffle(x_pos)

    if align:
        y_pos = x_pos
    else:
        y_pos = list(range(len(things)))
    random.shuffle(y_pos)

    relation = {}
    pos = ["right", "behind", "front", "left"]
    for p in pos:
        relation[p] = []
    for i in range(len(things)): 
        i_x_pos = x_pos.index(i)
        i_y_pos = y_pos.index(i)
        relation["right"].append(x_pos[i_x_pos + 1:])
        relation["left"].append(x_pos[:i_x_pos])
        relation["front"].append(y_pos[i_y_pos + 1:])
        relation["behind"].append(y_pos[:i_y_pos])
    
    return relation

def gen_data_point(things_bank, unique_things_num, max_same_things_num, total_things_num, unique_list=None, align=False):
    
    if type(unique_list) == type(None): 
        things = things_bank.get_things(unique_things_num, max_same_things_num, total_things_num)
    else:
        things = things_bank.get_things_by_uniq_list(unique_list)
    relations = get_space_relations(things, align)

    data = {}
    data["objects"] = things
    data["relationships"] = relations
    return data
    
if __name__ == "__main__":

    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    
    scene_file_name = "img_test_4_1_1_1_training.json"
    scene_file_path = os.path.abspath(os.path.join(raw_path, scene_file_name))

    # settings 
    unique_things_num = 3
    max_same_things_num = 3
    total_things_num = 7
    unique_list = [4,1,1,1]
    scene_num = 30
    align = False

    config = get_config()
    things_bank = ThingsBank(config)
    scenes = []
    for ct in range(scene_num):
        scenes.append(gen_data_point(things_bank, unique_things_num, max_same_things_num, total_things_num, unique_list=unique_list))

    with open(scene_file_path, 'w') as scene_file:
        json.dump(scenes, scene_file, indent=True)