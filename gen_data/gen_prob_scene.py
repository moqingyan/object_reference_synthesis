import json
import os
import sys
import random 
from functools import reduce 
import numpy as np

common_path = os.path.abspath(os.path.join(__file__, "../../src/common"))
src_path = os.path.abspath(os.path.join(__file__, "../../src"))
sys.path.append(common_path)
sys.path.append(src_path)

from utils import get_config

class ThingsBank():

    def __init__(self, config):
        self.config = config

    def get_prob(self, operation, max_prob):

        # helper function to generate n numbers which sums to 1
        def gen_attr(n, max_prob):
            values = [] 
            for i in range(n-1): 
                values.append(random.random())

            values = [ (value / sum(values)) * (1-max_prob) for value in values]
            values.append(max_prob)
            random.shuffle(values)
            return values

        num = len(self.config['choices'][operation])
        probs = gen_attr(num, max_prob)
        attrs = []
        
        for op in self.config['choices'][operation]:
            attrs.append(op)
        
        res = {}
        res["probs"] = probs
        res["attrs"] = attrs
        return res

    def get_thing(self, unsure_attr_num = 2, max_sure_prob = 1.0, max_unsure_prob=0.5):

        thing = {}
        sure = [ max_sure_prob for _ in range(len(self.config["choices"]) - unsure_attr_num)]
        unsure = [max_unsure_prob for _ in range(unsure_attr_num)]
        max_probs = sure + unsure
        random.shuffle(max_probs)

        for max_prob, (attr, choices) in zip(max_probs, self.config["choices"].items()):
            thing[attr] = self.get_prob(attr, max_prob)
        return thing

    def get_things(self, one_unsure_num, two_unsure_num, things_num):
        things = []
        
        for _ in range(one_unsure_num):
            things.append(self.get_thing(unsure_attr_num=1))

        for _ in range(two_unsure_num):
            things.append(self.get_thing(unsure_attr_num=2))

        for _ in range(things_num - one_unsure_num - two_unsure_num):
            things.append(self.get_thing(unsure_attr_num=0))

        return things

def get_space_relations(things):

    x_pos = list(range(len(things)))
    y_pos = list(range(len(things)))

    random.shuffle(x_pos)
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

def gen_data_point(things_bank, one_unsure_num, two_unsure_num, things_num):
    
    things = things_bank.get_things(one_unsure_num, two_unsure_num, things_num)
    relations = get_space_relations(things)

    data = {}
    data["objects"] = things
    data["relationships"] = relations
    return data

if __name__ == "__main__":

    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    
    scene_file_name = "7_prob_u1_3_u2_1.json"
    scene_file_path = os.path.abspath(os.path.join(raw_path, scene_file_name))

    # settings 
    things_num = 7
    scene_num = 1
    one_unsure_num = 3
    two_unsure_num = 1

    config = get_config()
    things_bank = ThingsBank(config)
    scenes = []
    for ct in range(scene_num):
        scenes.append(gen_data_point(things_bank,  one_unsure_num, two_unsure_num, things_num ))

    with open(scene_file_path, 'w') as scene_file:
        json.dump(scenes, scene_file, indent=True)

    