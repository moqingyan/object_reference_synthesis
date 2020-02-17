import os
import sys
import json 
from time import time

common_path = os.path.abspath(os.path.join(__file__, "../../common"))
print(common_path) 
sys.path.append(common_path)

from interpreter import synthesize, eu_synthesize
from query import SceneInterp
from utils import get_config, CENTER_RELATION
from cmd_args import cmd_args, logging
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


# mapping from num -> clause ? - seperate the clause rank from the search space 
# enumerative the clause combination --> each combination will lead to a state, how to save the states? 
# EuSolver based Enumerative search? 
# Look up table? 

def check_success(binding_dict, target):

    if not "var_0" in binding_dict.keys():
        return False

    if ((target in binding_dict["var_0"])  or (str(target) in binding_dict["var_0"])) and (len(binding_dict["var_0"]) == 1):
        return True

    return False 

def dfs_prog(scene, config, target, max_depth):
    interpreter = SceneInterp(scene, config)
    clean_state = interpreter.get_init_state()
    prog = synthesize(interpreter.interpreter, target, clean_state, is_uncertain=False, max_depth=max_depth)
    return prog

def eu_solve_prog(scene, config, target, max_depth):
    interpreter = SceneInterp(scene, config)
    clean_state = interpreter.get_init_state()
    prog = eu_synthesize(interpreter.interpreter, target, clean_state, is_uncertain=False, max_depth=max_depth)
    return prog

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
    max_depth = 10
    clauses_gen_class = EnumComb(max_depth, config)

    cts = []
    progs = []
    suc = 0
    total = 0
    
    scene_path = os.path.join(os.path.abspath(__file__ + "../../../../data/processed_dataset/raw/"), "3_1_1_1_1_testing.json")
    with open (scene_path, "r") as scene_file:
        scenes = json.load(scene_file)
    
    start = time()
    for sct, scene in enumerate(scenes):
        print(f"scene ct: {sct}")
        logging.info(f"scene ct: {sct}")
        for target in range(len(scene["objects"])):
            logging.info(f"total ct: {total}")
            total += 1
            
            # prog = dfs_prog(scene, config, target, max_depth)
            prog = eu_solve_prog(scene, config, target, max_depth)
            
            logging.info(prog)
            print( f"suc {not type(prog) == type(None)}")
            
            end = time()
            print (f"time used {end - start}")
            logging.info(f"time used {end - start}")

            if not type(prog) == type(None):
                # print(prog)
                suc += 1
                continue

            
            
    logging.info(f"suc: {suc}")
    end = time()
    print (f"time used {end - start}")
    logging.info(f"time used {end - start}")
    
    # print(cts)
    # print(progs)