import sys
import os 
import json 
import pickle 
from cmd_args import cmd_args
import math
import torch
import torch.nn as nn
from dataclasses import dataclass, astuple
from utils import NodeType, EdgeType, CENTER_RELATION, get_all_clauses, get_config
from graph_lib import BaseDiGraph

class Graph:

    def __init__(self, config, scene, target_id, max_var_num = cmd_args.max_var_num, hidden_dim = cmd_args.hidden_dim, thresholds = cmd_args.category_offsets, ground_truth_scene=None):
        self.config = config 
        self.scene = scene
        self.target_id = target_id
        self.thresholds = thresholds

        self.obj_num = len(self.scene['objects'])
        self.max_var_num = max_var_num

        self.base_graph = BaseDiGraph() 
        self.setup_graph_dict()
        self.ground_truth_scene = ground_truth_scene

    def prob_to_cat(self, prob):
        for ct, offset in enumerate(self.thresholds):
            if prob > offset:
                continue
            break
        return ct

    def setup_graph_dict(self):

        self.base_graph.add_vertex("target")
        
        # Construct attributes as nodes
        self.attrs = []
        for attrs in self.config["choices"].values():
            for attr in attrs:
                self.attrs.append(attr)
                self.base_graph.add_vertex(attr)

        # Construct prob center nodes
        self.prob_centers = [ f"cat_{cat}" for cat in list(range(len(self.thresholds))) ]

        for prob_center in self.prob_centers:
            self.base_graph.add_vertex(prob_center)

        self.prob_cats = []
        prob_count = 0

        # Construct object as object nodes
        for obj_id, obj in enumerate(self.scene['objects']):
            
            self.base_graph.add_vertex(f"obj_{obj_id}")

            for feature, value in obj.items():
                attrs = value["attrs"]
                probs = value["probs"]
            
                # add in edges between the object and its attributes
                for attr, prob in zip(attrs, probs):

                    cat = self.prob_to_cat(prob)
                    # ignore the low prob attributes
                    if cat == 0:
                        continue
                    
                    self.base_graph.add_vertex(f"prob_{prob_count}")
                    self.base_graph.add_undirected_edge(f"prob_{prob_count}", f"obj_{obj_id}",  EdgeType.edge_attr)
                    self.base_graph.add_undirected_edge(f"prob_{prob_count}", attr, EdgeType.center_prob)
                    self.base_graph.add_undirected_edge(f"prob_{prob_count}", f"cat_{cat}", EdgeType.center_prob)
                    self.prob_cats.append(f"prob_{prob_count}")
                    prob_count += 1
                    
            if obj_id == self.target_id:
                self.base_graph.add_undirected_edge("target", f"obj_{obj_id}", EdgeType.edge_attr)

        # Construct relation and relation center as rela-center nodes
        # self.relations = []
        self.rela_centers = ["center_right", "center_behind"]

        for rela_center in self.rela_centers:
            self.base_graph.add_vertex(rela_center)
        
        self.relations = []
        relation_count = 0
        for relation in self.scene['relationships']:
            # We only care about right and behind, since the inverse is obvious
            if relation == "right" or relation == "behind":

                if relation == "right":
                    rela_center = "center_right"
                if relation == "behind":
                    rela_center = "center_behind"

                for object_id, related_objects in enumerate(self.scene["relationships"][relation]):
                    for related_object_id in related_objects:
                        self.base_graph.add_vertex(f"relation_{relation_count}")
                        self.base_graph.add_undirected_edge(f"relation_{relation_count}", rela_center, EdgeType.center_relation)
                        self.base_graph.add_undirected_edge(f"relation_{relation_count}", f"obj_{object_id}", EdgeType.first)
                        self.base_graph.add_undirected_edge(f"relation_{relation_count}", f"obj_{related_object_id}", EdgeType.second)
                        self.relations.append(f"relation_{relation_count}")
                        relation_count += 1

        self.attr_or_rela = self.rela_centers + self.attrs

        # add variables
        self.vars = []
        binding_node_ct = 0
        for var_id in range(self.max_var_num):
            self.vars.append(f"var_{var_id}")
            self.base_graph.add_vertex(f"var_{var_id}")

            if cmd_args.binding_node:
                for target in range(self.obj_num):
                    self.base_graph.add_vertex(f"binding_{binding_node_ct}")
                    self.base_graph.add_undirected_edge(f"var_{var_id}", f"binding_{binding_node_ct}", EdgeType.binding)
                    self.base_graph.add_undirected_edge(f"obj_{target}", f"binding_{binding_node_ct}", EdgeType.binding)
                    binding_node_ct += 1
            else:
                for target in range(self.obj_num):
                    self.base_graph.add_undirected_edge(f"var_{var_id}", f"obj_{target}", EdgeType.binding)
                    binding_node_ct += 1

        # add global nodes
        if cmd_args.global_node:
            all_nodes = self.base_graph.get_names()
            self.base_graph.add_vertex("gb")
            for node in all_nodes:
                self.base_graph.add_edge(node, "gb", EdgeType.gb)

    def update_binding(self, binding_dict):
        names = self.base_graph.get_names()
        current_bindings = list(filter(lambda x: "binding" in x, names))
        
        for bd in current_bindings:
            neighbors = self.base_graph.get_neighbour(bd)
            for neighbor in neighbors:
                if "var" in neighbor:
                    var = neighbor
                if "obj" in neighbor:
                    obj = neighbor

            if not var in binding_dict:
                continue

            if not ((var, obj) in binding_dict.items()):
                self.base_graph.remove_vertex(bd)

    def get_nodes(self):
        return self.base_graph.get_names()

    def get_edge_info(self):
        return self.base_graph.edges()

    def get_attr_locs(self):
        names = self.base_graph.get_names()
        idxes = []
        for idx, name in enumerate(names):
            if name in self.attrs:
                idxes.append(idx)
        return idxes

    def get_rela_locs(self):
        names = self.base_graph.get_names()
        idxes = []
        for idx, name in enumerate(names):
            if name in self.rela_centers:
                idxes.append(idx)
        return idxes

    def get_attr_or_rela_locs(self):
        names = self.base_graph.get_names()
        idxes = []
        for idx, name in enumerate(names):
            if name in self.attr_or_rela:
                idxes.append(idx)
        return idxes

    def get_var_locs(self, rel=None):
        names = self.base_graph.get_names()
        if not type(rel) == type(None):
            var = rel
        else:
            var = self.vars

        idxes = []
        for idx, name in enumerate(names):
            if name in var:
                idxes.append(idx)
        return idxes

    def get_attr_by_idx(self, idx):
        return self.attrs[idx]

    def get_var_by_idx(self, idx):
        return self.vars[idx]

    def get_rela_by_idx(self, idx):
        rela = self.rela_centers[idx]
        if "right" in rela:
            return "right"
        return "behind"

    def get_attr_or_rela_by_idx(self, idx):
        attr_or_rela = self.attr_or_rela[idx]
        if "right" in attr_or_rela:
            return "right"
        if "behind" in attr_or_rela:
            return "behind"
        return attr_or_rela
    
if __name__ == "__main__":

    # We use a configuration file to specify the edges and nodes choices
    config = get_config()
    graph_file_name = "prob_unit_test_2.pkl"
    scene_file_name = "prob_unit_test_2.json"

    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    
    graphs_path = os.path.join(raw_path, graph_file_name)
    scenes_path = os.path.abspath(os.path.join(raw_path, scene_file_name))

    # In the pytorch geometry package, only int and tensor seems to be allowed to save
    # we process all the graphs and save them to a file.
    with open(scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)
    
    graphs = []

    for scene in scenes:
        for target_id in range(len(scene["objects"])):
            graph = Graph(config, scene, target_id)
            graphs.append(graph)
    
    edge_info = graph.get_edge_info()
    
