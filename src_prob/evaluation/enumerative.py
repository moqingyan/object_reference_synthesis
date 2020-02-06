import os
import sys
import json 

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
print(common_path)
sys.path.append(common_path)

from query import query
from utils import get_config, CENTER_RELATION
from cmd_args import cmd_args
import itertools 

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

class EnumComb():

    def __init__(self, max_depth, config):
        self.max_depth = max_depth
        self.enum_class = EnumClause(config)
        self.config = config

    def get_clauses(self):
        
        for depth in range(self.max_depth+1):
            clause_generator = self.enum_class.get_clause() 
            gen_clauses = itertools.product(clause_generator, repeat=depth)
            for clauses in gen_clauses:
                yield list(clauses)

def check_success(binding_dict, target):

    if not "var_0" in binding_dict.keys():
        return False

    if binding_dict["var_0"] == [str(target)]:
        return True

    return False 

def find_prog(scene, target, clauses_gen_class):

    prog_generator = clauses_gen_class.get_clauses()
    for ct, prog in enumerate(prog_generator):
        if prog == []:
            continue
            
        binding_dict = query(scene, prog, clauses_gen_class.config)
        suc = check_success (binding_dict, target)
        if suc:
            return ct, prog
    
    print("Failed to find")
    return ct, None

def test_EnumClause(config):

    clause_gen = EnumClause(config)
    a = clause_gen.get_clause()
    b = clause_gen.get_clause()
    
    print (a)
    print (b)

    for i, c in enumerate(a):
        print (c)
        if i > 2:
            break

    for i, c in enumerate(b):
        print (c)
        if i > 2:
            break

def test_EnumComb(max_depth, config):
    clauses_gen = EnumComb(max_depth, config)
    a = clauses_gen.get_clauses()
    for i, c in enumerate(a):
        print(c)

if __name__ == "__main__":
    config = get_config()
    max_depth = 2
    clauses_gen_class = EnumComb(max_depth, config)

    cts = []
    progs = []
    suc = 0
    
    scene_path = os.path.join(os.path.abspath(__file__ + "../../../../data/processed_dataset/raw/"), cmd_args.scene_file_name)
    with open (scene_path, "r") as scene_file:
        scenes = json.load(scene_file)
    
    for sct, scene in enumerate(scenes):
        print(f"scene ct: {sct}")
        for target in range(len(scene["objects"])):
            print(f"target: {target}")
            ct, prog = find_prog(scene, target, clauses_gen_class)
            cts.append(ct)
            progs.append(prog)

            if not type(prog) == None:
                print(prog)
                suc += 1
                continue

    print(cts)
    print(progs)