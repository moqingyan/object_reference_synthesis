import os
import sys
import json 
import re 

def analysis_greedy_prog(prog_str):
    clause_res = re.findall("[a-z]*\(.*?\)", prog_str)
    clause_num = len(clause_res)
    var_res = re.findall("_.[0-9]|x", prog_str)
    var_num = len(set(var_res))
    relation_res = re.findall("left|right|behind|front", prog_str)
    relation_num = len(relation_res)
    if prog_str == "x <- ":
        correct = 0
    else:
        correct = 1
    return (var_num, relation_num, clause_num, correct)
    
def analysis_cont_prog(clauses):
    clause_num = len(clauses)
    var_res = re.findall("[0-9]+", str(clauses))
    var_num = len(set(var_res))
    relation_res = re.findall("left|right|behind|front", str(clauses))
    relation_num = len(relation_res)
    return (var_num, relation_num, clause_num, 1)

if __name__ == '__main__':

    # arrange all the directories
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../../data"))

    scene_file_name = "img_test_500.json"
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, scene_file_name))
    greedy_res_path = os.path.abspath(os.path.join(data_dir, "./eval_result/greedy_res.json"))
    cont_dir = os.path.abspath(os.path.join(data_dir, "./eval_result/cont_overfit_res"))

    with open(greedy_res_path, 'r') as greedy_res_file:
        greedy_res = json.load(greedy_res_file)

    with open (scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)

    problems = []
    for scene in scenes:
        for obj in scene["objects"]:
            problems.append((scene, obj))
    
    cont_analysises = []
    greedy_analysises = []
    for prob_id, problem in enumerate(problems):
        with open (os.path.join(str(cont_dir), str(prob_id)), 'r') as cont_res_file:
            clauses = json.load(cont_res_file)

        cont_analysis = analysis_cont_prog(clauses[0])
        greedy_analysis = analysis_greedy_prog(greedy_res[str(prob_id)][0])

        cont_analysises.append(cont_analysis)
        greedy_analysises.append(greedy_analysis)

    same_out = 0
    
    # for prob_id, (greedy_analysis, cont_analysis) in enumerate(zip(greedy_analysises, cont_analysises)):
    #     if greedy_analysis == cont_analysis:
    #         same_out += 1
    #     if greedy_analysis[3] == 0:
    #         print (f"prob_id: {prob_id}, var num: {cont_analysis[0]}, relation_num: {cont_analysis[1]}, clause_num: {cont_analysis[2]}")
    # print(same_out)

    prob_ids = [519]
    for prob_id in prob_ids:
        scene, obj = problems[prob_id]
        image_idx = scene["image_index"]
        tar_obj = obj
        
        print (f"picture idx: {image_idx}")
        print (f"target object: {tar_obj}")

    

    
    
