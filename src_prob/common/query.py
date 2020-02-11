import os
import sys
import time
import subprocess
import re
import json
import socket

interpreter_path = os.path.abspath(os.path.join(__file__, "../../../dragoman/interpret"))
from cmd_args import cmd_args, logging
from utils import CENTER_RELATION, get_operands, get_config
import numpy as np
from interpreter import GroundTuples, BinaryClause, UnaryClause, Program, Interpreter, execute, synthesize

def matrix2tps(mat):
    tps = []
    for ct, row in enumerate(mat):
        for e in row:
            tps.append((ct, e))
    return tps

def get_attr_operation(name, configure):
    for key, value in configure["choices"].items():
        if name in value:
            return key
    return


class SceneInterp():
    def __init__(self, scene, config, is_uncertain = cmd_args.prob_dataset):
        # self.scene = scene
        self.config = config
        self.is_uncertain = is_uncertain
        self.ground_certain, self.ground_uncertain = self.scene2ground(scene)

        self.unary_names = self.ground_certain.unary.keys()
        self.binary_names = self.ground_certain.binary.keys()
        self.interpreter = Interpreter(self.ground_certain, self.ground_uncertain, cmd_args.max_var_num, len(scene["objects"]))
        
    def get_init_state(self):
        return self.interpreter.reset()

    # TODO: we can save ground instead of scene to query to save time
    def scene2ground_certain(self, scene):
        unary, binary = get_operands()
        if self.is_uncertain:
            return GroundTuples(unary, binary)

        for ct, thing in enumerate(scene["objects"]):
            for feature, value in thing.items():
                if type(value) == list:
                    continue
                if value in unary.keys():
                    unary[value].append(ct)

        for rela_op, order_mat in scene["relationships"].items():
            if rela_op in binary.keys():
                binary[rela_op] = matrix2tps(order_mat)

            pass #leave the prob model to implement (threshold?)

        return GroundTuples(unary, binary)

    def scene2ground_uncertain(self, scene, threshold=0.5):
        unary, binary = get_operands()

        if not self.is_uncertain:
            return GroundTuples(unary, binary)

        for ct, object_ in enumerate(scene["objects"]):
            for feature, value in object_.items():
                if type(value) == dict:
                    attrs = value["attrs"]
                    probs = value["probs"]
                
                    for attr, prob in zip(attrs, probs):
                        if attr in unary.keys() and prob > threshold:
                            unary[attr].append(ct)
                else: 
                    if type(value) == list:
                        continue
                    if value in unary.keys():
                        unary[value].append(ct)

        for rela_op, order_mat in scene["relationships"].items():
            if rela_op in binary.keys():
                binary[rela_op] = matrix2tps(order_mat)
      
        return GroundTuples(unary, binary)

    def scene2ground(self, scene):
        return self.scene2ground_certain(scene), self.scene2ground_uncertain(scene)

    def clause_transform(self, clause):

        if clause[0] in self.binary_names:
            
            if type(clause[1]) == str and "var" in clause[1]:
                var1 = int(clause[1][4:])
                var2 = int(clause[2][4:])
            else:
                var1 = int(clause[1])
                var2 = int(clause[2])
            return BinaryClause(clause[0],var1, var2)
        if clause[1] in self.unary_names:
            if type(clause[2]) == str and "var" in clause[2]:
                var1 = int(clause[2][4:])
            else:
                var1 = int(clause[2])
            return UnaryClause(clause[1], var1)
        raise Exception("clause not found")

    def clauses2prog(self, clauses):
        clauses = [self.clause_transform(clause) for clause in clauses]
        return Program(clauses)

    def process_binding_to_dict(self, bindings):
        var_num = cmd_args.max_var_num
        binding_dict = {}
        for ct in range(var_num):
            binding_dict[f"var_{ct}"] = set()

        for var_id, binding in enumerate(bindings):
            for obj_id, obj in enumerate(binding):
                if obj == 1:
                    binding_dict[f"var_{var_id}"].add(obj_id)

        return binding_dict

    def fast_query(self, clauses):

        program = self.clauses2prog(clauses) # Time consuming
        state = execute(self.interpreter, program, is_uncertain=self.is_uncertain) # Not using execute function

        bindings = self.interpreter.get_marginal_bindings(state, is_uncertain=self.is_uncertain)
        binding_dict = self.process_binding_to_dict(bindings.marginal_binding)

        # print(bindings.marginal_binding)
        return binding_dict

    def state_query(self, state, new_clause):
        clause =  self.clause_transform(new_clause)
        new_state = self.interpreter.step(state, clause, self.is_uncertain)
        bindings = self.interpreter.get_marginal_bindings(new_state, self.is_uncertain)
        binding_dict = self.process_binding_to_dict(bindings.marginal_binding)

        # print(bindings.marginal_binding)
        return binding_dict, new_state

    def useful_check(self, state, new_clause):
        clause =  self.clause_transform(new_clause)
        useful = self.interpreter.is_clause_useful(state, clause, is_uncertain=self.is_uncertain)
        return useful 

    def interp_enum(self, clauses, target, max_depth=1):

        # Given a good candidate of a partial program, exploit the clauses
        program = self.clauses2prog(clauses) # Time consuming
        state = execute(self.interpreter, program, is_uncertain=self.is_uncertain) # Not using execute function
        prog = synthesize(self.interpreter, target, state, is_uncertain=self.is_uncertain, max_depth=max_depth)

        if type(prog) == type(None):
            return

        new_prog = []
        for clause in prog.clauses:
            new_clause = clause.to_tuple()
            if len(new_clause) == 2:
                op = get_attr_operation(new_clause[0], self.config)
                new_clause = [op, new_clause[0], new_clause[1]]
            new_prog.append(new_clause)

        return new_prog

if __name__ == "__main__":
    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    scenes_path = os.path.abspath(os.path.join(raw_path, "prob_unit_test_2.json"))

    with open(scenes_path) as scenes_file:
        scenes = json.load(scenes_file)
    scene = scenes[0]
    config = get_config()

    scene_interp = SceneInterp(scene, config)
    state = scene_interp.get_init_state()
    ground = scene_interp.ground_uncertain

    clauses = [['size', 'large', 'var_0'], ['material', 'metal', 'var_2'], ['size', 'large', 'var_2'], ['shape', 'cylinder', 'var_2']]
    clause_1 = ['color', 'brown', 'var_0']
    clause_2 = ['color', 'purple', 'var_0']
    prog = scene_interp.clauses2prog(clauses)
    print(prog)

    state = scene_interp.get_init_state()
    binding_dict_1, state = scene_interp.state_query(state, clause_1)
    binding_dict_2, state = scene_interp.state_query(state, clause_2)