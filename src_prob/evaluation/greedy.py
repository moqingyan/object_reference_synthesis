import os
import sys
import json
import time

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
print(common_path)
sys.path.append(common_path)

from query import query
from utils import get_config, CENTER_RELATION
from cmd_args import cmd_args

class EnumClause():

    def __init__(self, config):
        self.config = config

    def get_clause(self):
        
        for op in (self.config["edge_types"]["attributes"]):
            for attr in self.config["choices"][op]:
                for var in range(cmd_args.max_var_num):
                    yield [op, attr, var]

        for op in CENTER_RELATION:
            for var1 in range(cmd_args.max_var_num):
                for var2 in range(cmd_args.max_var_num):
                    if not var1 == var2:
                        yield [op, var1, var2]

def check_success(binding_dict, target):

    if not "var_0" in binding_dict.keys():
        return False

    if binding_dict["var_0"] == [str(target)]:
        return True

    return False 

def check_possible(binding_dict, target):

    if not "var_0" in binding_dict.keys():
        return True 

    if str(target) not in binding_dict["var_0"] :
        return False #

    return True

class Greedy():

    def __init__(self, config,  max_depth=10): 
        self.max_depth = max_depth
        self.config = config
        self.clause_generator = EnumClause(config)

    def get_option_num(self, clauses, scene, target):
        binding_dict = query(scene, clauses, self.clause_generator.config)
        success = check_success(binding_dict, target)
        possible = check_possible(binding_dict, target)
        option_num = len(binding_dict["var_0"]) if "var_0" in binding_dict.keys() else len(scene["objects"])
        return option_num, success, possible

    def get_clauses(self, scene, target):
        self.current_clauses = []
        self.current_binding_num = len(scene["objects"])

        while ( len(self.current_clauses) <= self.max_depth ):
            print(f"Current updating:{len(self.current_clauses)}")
            max_clause_info = None

            for clause in self.clause_generator.get_clause():

                clauses = self.current_clauses + [clause]
                option_num, success, possible = self.get_option_num(clauses, scene, target)
                
                if not possible:
                    continue

                if success:
                    return clause

                info_diff = self.current_binding_num - option_num
                if ((type(max_clause_info) == type(None)) or info_diff > max_clause_info[2]):
                    if not (type(max_clause_info) == type(None)):
                        print(info_diff)
                        print(max_clause_info[2])
                    max_clause_info = (clause, option_num, info_diff) 
                    print (f"Update to {max_clause_info}")
                    
            self.current_clauses += [max_clause_info[0]]
            self.current_binding_num = max_clause_info[1]
        
        

if __name__ == '__main__':

    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))

    scene_file_name = "img_test_500.json"
    greedy_res = "greedy_img_test_500.json"

    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, scene_file_name))
    greedy_path = os.path.abspath(os.path.join(data_dir, f"eval_result/{greedy_res}"))
    config = get_config()

    with open(scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)

    res = {}
    prob_id = 0
    start_time = time.time()

    # scene = {"objects": [{"size": "small", "color": "brown", "shape": "cylinder", "material": "metal"}, {"size": "small", "color": "brown", "shape": "cylinder", "material": "metal"}, {"size": "large", "color": "purple", "shape": "sphere", "material": "rubber"}, {"size": "large", "color": "gray", "shape": "cube", "material": "rubber"}, {"size": "small", "color": "blue", "shape": "cube", "material": "metal"}, {"size": "small", "color": "brown", "shape": "cylinder", "material": "metal"}, {"size": "small", "color": "gray", "shape": "cube", "material": "rubber"}], "relationships": {"right": [[5, 2, 1, 4], [4], [1, 4], [0, 5, 2, 1, 4], [], [2, 1, 4], [3, 0, 5, 2, 1, 4]], "behind": [[6, 3], [6, 3, 0, 5, 2], [6, 3, 0, 5], [6], [6, 3, 0, 5, 2, 1], [6, 3, 0], []], "front": [[5, 2, 1, 4], [4], [1, 4], [0, 5, 2, 1, 4], [], [2, 1, 4], [3, 0, 5, 2, 1, 4]], "left": [[6, 3], [6, 3, 0, 5, 2], [6, 3, 0, 5], [6], [6, 3, 0, 5, 2, 1], [6, 3, 0], []]}} 
    target = 0 
    
    for ct_scene, scene in enumerate(scenes):
        print(f"scene: {ct_scene}")
        for ct, obj in enumerate(scene["objects"]):
            print(f"target_id: {ct}")
            greedy_search = Greedy(config) 
            clause = greedy_search.get_clauses(scene, ct)
            print(f"prog is {clause}")

    # for ct_s, raw_scene in enumerate(raw_scenes):
    #     scene = Scene(raw_scene)
    #     for ct_o, obj in enumerate(scene.objects):
    #         prob_id += 1

    #         # if not (prob_id == 208):
    #         #     continue

    #         # try:
    #         print (f"scene count {ct_s}, obj count {ct_o}")
    #         solution = greedy (obj, scene)
    #         # except RecursionError:
    #         #     print (len(solution.clauses))
    #         #     solution = top()
    #         #     solution.timeout = True
    #         #     print (f"Fail to solve: {prob_id}")

    #         res[prob_id] = (str(solution), solution.timeout)
            
    #         if prob_id % 100 == 0:
    #             print (prob_id)
    #             with open(greedy_path, 'w') as greedy_file:
    #                 json.dump(res, greedy_file, indent=True)
                
            
    end_time = time.time()

    # with open(greedy_path, 'w') as greedy_file:
    #     json.dump(res, greedy_file, indent=True)

    print (f"time spent: {end_time - start_time}")
