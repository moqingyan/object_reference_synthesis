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

@dataclass
class GraphNode(object):

    def __init__(self, index, node_type, name):
        self.index = index
        self.node_type = node_type
        self.name = name

class Edge(object):

    def __init__(self, src_idx, dst_idx, edge_type):
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.edge_type = edge_type

    def to_tuple(self, cat = 5):
        return (self.src_idx, self.dst_idx, self.edge_type.value)

    def __eq__(self, other):
        if self.dst_idx == other.dst_idx and self.src_idx == other.src_idx and self.node_type == other.node_type:
            return True

    def __ne__(self, other):
        return not self.__eq__(other)

class Graph2(object):
    
    def __init__(self, config, scene, target_id, offsets = cmd_args.category_offsets, max_var_num = cmd_args.max_var_num, hidden_dim = cmd_args.hidden_dim):

        self.config = config 
        self.scene = scene
        self.target_id = target_id
        self.offsets = offsets
        self.obj_num = len(self.scene['objects'])
        
        self.nodes = []
        self.edges = []

        self.max_var_num = max_var_num
        
        self.cats = {}
        self.add_cats()

        self.target_idx = None
        self.add_target()

        self.attrs = {}
        self.add_attributes()

        self.objs = {}
        for obj_ct, object_info in enumerate(scene['objects']):
            self.add_object(object_info, obj_ct)
        
        self.relations = {}
        self.add_relations()

        self.vars = {}
        self.add_variables()
        self.add_attr_or_rela()

    def get_attr_of(self, obj_id):
        attrs = []
        attrs_id = []
        obj_idx = self.objs[obj_id]

        for edge in self.edges:
            if edge.src_idx == obj_idx:
                if (self.node[edge.dst_idx].node_type == NodeType.attr) or (self.node[edge.dst_idx].node_type == NodeType.target):
                    attrs.append(self.node[edge.dst_idx].name)
                    attrs_id.append(edge.dst_idx)
        return attrs, attrs_id

   
    def add_node(self, node_type, name):
        index = len(self.nodes)
        self.nodes.append(GraphNode(index, node_type, name))
        return index

    def add_undirected_edge(self, src_idx, dst_idx, edge_type):
        assert (type(src_idx) == int and type(dst_idx) == int )
        self.edges.append(Edge(src_idx, dst_idx, edge_type))
        self.edges.append(Edge(dst_idx, src_idx, edge_type))

    def remove_undirected_edge(self, src_idx, dst_idx, edge_type):

        def same(edge):
            if edge.src_idx == src_idx and edge.dst_idx == dst_idx:
                return True 
            if edge.src_idx == dst_idx and edge.dst_idx == src_idx:
                return True
            return False 

        self.edges = list(filter(lambda edge: not same(edge), self.edges))

    def prob_to_cat(self, prob):
        for ct, offset in enumerate(self.offsets):
            if prob > offset:
                continue
            break
        return ct

    def add_cats(self):
        for offset in self.offsets:
            cat = self.prob_to_cat(offset)
            cat_idx = self.add_node(NodeType.prob_category, f"cat_{cat}")
            self.cats[cat] = cat_idx

    def add_attributes(self):
        node_attrs = []
        for v in self.config["choices"].values():
            node_attrs += v
        for idx, attr in enumerate(node_attrs):
            self.attrs[attr] = self.add_node(NodeType.attr, attr)

    def add_target(self):
        self.target_idx = self.add_node(NodeType.target, "target")

    # adding the object itself and its attribute to the graph
    def add_object(self, object_info, object_id):


        object_idx = self.add_node(NodeType.obj, f"obj_{object_id}")

        self.objs[object_id] = (object_idx)
        obj_attrs_dict = [object_info[key] for key in self.config["choices"].keys()]

        # attach the attribute node to the graph node
        for op_info in obj_attrs_dict:
            for i, attr in enumerate(op_info['attrs']): 
                attr_prob = op_info["probs"][i]
                attr_cat = self.prob_to_cat(attr_prob)

                attr_cat_idx = self.cats[attr_cat]
                prob_cat_type = EdgeType.prob_cat_1 if (attr_cat_idx == 0) else (EdgeType.prob_cat_2 if (attr_cat_idx == 1) else EdgeType.prob_cat_3)

                attr_idx = self.attrs[attr]
                if attr_prob > 0:
                    self.add_undirected_edge(attr_idx, attr_cat_idx, EdgeType.edge_attr)
                    self.add_undirected_edge(attr_cat_idx, object_idx, EdgeType.edge_attr)
                    
                    # Using two edges to connect the nodes?
                    # self.add_undirected_edge(attr_idx, object_idx, EdgeType.prob_category)
                    # self.add_undirected_edge(attr_idx, object_idx, attr_prob)

                    # self.add_undirected_edge(attr_idx, object_idx, EdgeType.edge_attr)

        # add the target label to the target node
        if object_id == self.target_id:
            attr = "target"
            self.add_undirected_edge(self.target_idx, object_idx, EdgeType.edge_attr)

    # add the variable nodes and their bondings to the graph
    def add_variables(self):

        for var_id in range(self.max_var_num):
            # add a new node for var
            var_idx = self.add_node(NodeType.var, f"var_{var_id}")
            self.vars[var_id] = var_idx

            # add bindings
            for target in range(self.obj_num):
                target_idx = self.objs[target]
                self.add_undirected_edge(var_idx, target_idx, EdgeType.bonding)

    # adding the edges between objects
    def add_relations(self):

        self.rela_center = dict()

        # construct two centrual nodes to pass information
        for relation in CENTER_RELATION:
            rela_center_idx = self.add_node (NodeType.center_relation, relation)
            self.rela_center[relation] = rela_center_idx
            self.relations[relation] = []

        for relation in self.scene['relationships']:
            # We only care about right and behind, since the inverse is obvious
            if relation == "right" or relation == "behind":

                for object_id, related_objects in enumerate(self.scene["relationships"][relation]):
                    for related_object_id in related_objects:
                        
                        fst_idx = self.objs[object_id]
                        snd_idx = self.objs[related_object_id]
                        rela_idx = self.add_node(NodeType.relation, relation)
                        center_idx = self.rela_center[relation]
                        
                        self.add_undirected_edge(rela_idx, center_idx, EdgeType.center_relation)
                        self.add_undirected_edge(rela_idx, fst_idx, EdgeType.first)
                        self.add_undirected_edge(rela_idx, snd_idx, EdgeType.second)

                        self.relations[relation].append(rela_idx)

    def add_attr_or_rela(self):
        self.attr_or_rela = {}
        self.attr_or_rela.update(self.attrs)
        self.attr_or_rela.update(self.rela_center)

    def get_edge_info(self):
        edge_infos = [ edge.to_tuple() for edge in self.edges]
        first, second, edge_type = zip(*edge_infos)
        return [ list(first), list(second)], list(edge_type)

    def update_binding(self, bonding_dict):

        # go through all the edges, remove all the bonding edge that of "bonding" type,
        # but not in the bonding_dict
        # (src_idx, dst_idx, edge_type, edge_attr, edge_embedding)
        to_remove = []
        for edge in self.edges:
            if edge.edge_type == EdgeType.bonding:

                var_name = self.nodes[edge.src_idx].name 
                obj_name = self.nodes[edge.dst_idx].name
                obj_name = obj_name.split('_')[1]

                # if the variable is not bond to anything, it is not mentioned in the
                # clauses, ignore the case
                if var_name not in bonding_dict.keys():
                    continue
            
                # if the edge does not exists in the dict, delete the edge.
                if obj_name not in bonding_dict[var_name]:
                    to_remove.append(edge)

        for edge in to_remove:
            self.remove_undirected_edge(edge.src_idx, edge.src_idx, EdgeType.bonding)
        
        # if not len(to_remove) == 0:
        #     print(f"removed {len(to_remove)} in the graph")
    
    # def get_node_embedding(self, node_name):

    #     if "var_" in node_name:
    #         embedding = self.var_encoder(torch.tensor(node_name.split("_")[1]))
    #     elif "obj_" in node_name:
    #         embedding = self.obj_encoder(torch.tensor(node_name.split("_")[1]))
    #     else:
    #         embedding = self.encoder(node_name)

    #     return embedding

    # def get_edge_embedding(self, edge_type):
    #     return self.encoder(edge_type)


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
    graph_file_name = "prob_CLEVR_val_unit_2.pkl"
    scene_file_name = "prob_CLEVR_val_unit_2.json"

    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    # graphs_path = os.path.join(raw_path, cmd_args.graph_file_name)
    # scenes_path = os.path.abspath(os.path.join(raw_path, cmd_args.scene_file_name))
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
    with open(graphs_path, 'wb') as graphs_file:
        pickle.dump(graphs, graphs_file) 
    
    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file)
