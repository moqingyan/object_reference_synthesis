import os
import sys 

import subprocess 
import re 
import json 
import socket  

interpreter_path = os.path.abspath(os.path.join(__file__, "../../../dragoman/interpret"))
from cmd_args import cmd_args, logging
from utils import PORT, LOCALHOST, CENTER_RELATION
import numpy as np
import time

# def interpret_json(json_path):
#     res = subprocess.run([interpreter_path, "-i", json_path],  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     res.check_returncode()
#     return res.stdout.decode("utf-8")
s = socket.socket()
s.connect((LOCALHOST, PORT)) 

def merge_results(res):

    res_dict = dict()
    res = res.split()
        
    #get the  variable names
    variables = res[0].split(",")

    if len(res) == 1:
        # not a valid choice
        for variable in variables:
            res_dict[variable] = []
    else:
        
        # values = np.array([[int(i) for i in pair.split(",")] for pair in res[1:]])
        # for idx, variable in enumerate(variables):
        #     res_dict[variable] = list(set(values[:, idx]))
        # values = [ set(values) for var_value in values ]
        values = list(zip(* [pair.split(",") for pair in res[1:]]))

        for idx, variable in enumerate(variables):
            res_dict[variable] = list(set(values[idx]))

    # print(res_dict)
    return res_dict

def clause2query(clause, config):

    query = dict()
    # the result we obtain from the NN:
    # operation_list = ["left", "right", "front", "back", "size", "color", "shape", "material"]
    
    if clause[0] in CENTER_RELATION:
        query["kind"] = "relate"
        query["relation"] = clause[0]
        query["left"] = clause[1] if "var" in str(clause[1]) else ("var_" + str(clause[1]))
        query["right"] = clause[2] if "var" in str(clause[2]) else ("var_" + str(clause[2]))
    else:
        query["kind"] = "select"
        query["attribute"] = clause[0]
        query["value"] = clause[1]
        query["variable"] = clause[2] if "var" in str(clause[2]) else ("var_" + str(clause[2]))

    # print(query)
    return query

def sample_scene(prob_scene):
    objects = []
    scene = {}
    for obj in prob_scene["objects"]:
        new_obj = {}
        for op, choices in obj.items():
            probs = choices["probs"]
            attrs = choices["attrs"] 
            sel_attr = np.random.choice(attrs, size=1, p=probs)[0]
            new_obj[op] = sel_attr
        objects.append(new_obj)
    
    scene["objects"] = objects
    scene["relationships"] = prob_scene["relationships"]
    return scene

def query(scene, clauses, config, json_path=cmd_args.json_path):
    request = dict()
    if cmd_args.prob_dataset:
        request["scene"] = sample_scene(scene)
    else:
        request["scene"] = scene
    request["clause"] = [ clause2query(clause, config) for clause in clauses]

    # with open(json_path, 'w') as json_file:
    #     json.dump(request, json_file)
    # res = interpret_json(json_path)
    # res_dict = merge_results(res)

    message = json.dumps(request) + "\n"
    # logging.info(message)
    s.send(message.encode())
    # receive data from the server 
    res = s.recv(16384).decode()
    r = res.split('\n')
    # logging.info(f"res: {r}")
    res_dict = merge_results(res)
    # logging.info(res_dict)
    
    return res_dict


if __name__ == "__main__":
    json_path = os.path.abspath(os.path.join(__file__, "../../../data/processed_dataset/raw/img_test_1_prob_things_all_certain.json"))
    scene = json.load(open(json_path))
    s = sample_scene(scene[0])
    res = interpret_json(json_path)
    res_dict = merge_results(res)

