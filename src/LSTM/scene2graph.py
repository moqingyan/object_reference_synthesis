import sys
import os 
import json 
import pickle 
from cmd_args import cmd_args
import torch
import torch.nn as nn
from dataclasses import dataclass, astuple
from utils import NodeType, EdgeType, CENTER_RELATION

@dataclass
class GraphNode(object):

    def __init__(self, index, node_type, name):
        self.index = index
        self.node_type = node_type
        self.name = name
        self.neighbors = []
        self.attached_edges = []

class Edge(object):

    def __init__(self, src_idx, dst_idx, edge_type):
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.edge_type = edge_type

    def to_tuple(self):
        return (self.src_idx, self.dst_idx, self.edge_type.value)

    def __eq__(self, other):
        if type(other) == type(None):
            return False 

        if self.dst_idx == other.dst_idx and self.src_idx == other.src_idx and self.edge_type == other.edge_type:
            return True
        
        return False

    def __ne__(self, other):
        return not self.__eq__(other)


class Graph(object):
    
    def __init__(self, config, scene, target_id, max_var_num = cmd_args.max_var_num, hidden_dim = cmd_args.hidden_dim):

        # This graph are seperated into two parts: 
        # 1. The ground truth part 
        # 2. The binding part
        
        self.config = config 
        self.scene = scene
        self.target_id = target_id
        self.obj_num = len(self.scene['objects'])
        
        self.nodes = []
        self.edges = []

        self.max_var_num = max_var_num
        
        self.target_idx = None
        self.add_target()

        self.attrs = dict()
        self.add_attributes()

        self.objs = dict()
        for obj_ct, object_info in enumerate(scene['objects']):
            self.add_object(object_info, obj_ct)
        
        self.relations = dict()
        self.add_relations()

        self.add_attr_or_rela()

        # All the edges between varables and objects are subjected to be removed 
        self.vo_edge_dict = {}
        self.init_vo_edge_dict()

        self.vars = {}
        self.add_variables()

    # def get_attr_of(self, obj_id):
    #     attrs = []
    #     attrs_id = []
    #     obj_idx = self.objs[obj_id]

    #     for edge in self.edges:
    #         if edge.src_idx == obj_idx:
    #             if (self.node[edge.dst_idx].node_type == NodeType.attr) or (self.node[edge.dst_idx].node_type == NodeType.target):
    #                 attrs.append(self.node[edge.dst_idx].name)
    #                 attrs_id.append(edge.dst_idx)
    #     return attrs, attrs_id

    def init_vo_edge_dict(self):
        for var_id in range(cmd_args.max_var_num):
            self.vo_edge_dict[var_id] = {}

    def add_node(self, index, node_type, name): 
        self.nodes.append(GraphNode(index, node_type, name)) 

    def add_edge(self, src_idx, dst_idx, edge_type):
        edge_index = len(self.edges)
        self.edges.append(Edge(src_idx, dst_idx, edge_type))
        self.nodes[src_idx].attached_edges.append(edge_index)
        self.nodes[src_idx].neighbors += [dst_idx]
        return edge_index 

    def add_undirected_edge(self, src_idx, dst_idx, edge_type):
        edge_idx_1 = self.add_edge(src_idx, dst_idx, edge_type)
        edge_idx_2 = self.add_edge(dst_idx, src_idx, edge_type)
        return edge_idx_1, edge_idx_2

    def remove_edge(self, edge_idx):
        
        edge = self.edges[edge_idx]
        if edge == None:
            return

        src_idx = edge.src_idx
        dst_idx = edge.dst_idx
        self.nodes[src_idx].neighbors.remove(dst_idx)
        self.nodes[src_idx].attached_edges.remove(edge_idx)
        self.edges[edge_idx] = None

    def add_attributes(self):

        start_idx = len(self.nodes)
        node_attrs = []

        for v in self.config["choices"].values():
            node_attrs += v

        for idx, attr in enumerate(node_attrs):
            self.attrs[attr] = start_idx + idx
            self.add_node(start_idx + idx, NodeType.attr, attr)

    def add_target(self):
        self.target_idx = len(self.nodes)
        self.add_node(self.target_idx, NodeType.target, "target")

    # adding the object itself and its attribute to the graph
    def add_object(self, object_info, object_id):
        object_idx = len(self.nodes)
        self.add_node(object_idx, NodeType.obj, f"obj_{object_id}")

        self.objs[object_id] = (object_idx)
        obj_attrs = [object_info[key] for key in self.config["choices"].keys()]

        # attach the attribute node to the graph node
        for attr in obj_attrs:
            attr_idx = self.attrs[attr]
            self.add_undirected_edge(attr_idx, object_idx, EdgeType.edge_attr)

        # add the target label to the target node
        if object_id == self.target_id:
            attr = "target"
            self.add_undirected_edge(self.target_idx, object_idx, EdgeType.edge_attr)

    # add the variable nodes and their bondings to the graph
    def add_variables(self):

        for var_id in range(self.max_var_num):
            # add a new node for var
            var_idx = len(self.nodes)
            self.vars[var_id] = var_idx
            self.add_node(var_idx, NodeType.var, f"var_{var_id}")

            # add bindings
            for target in range(self.obj_num):
                target_idx = self.objs[target]
                edge_idx_1, edge_idx_2 = self.add_undirected_edge(var_idx, target_idx, EdgeType.bonding)
                self.vo_edge_dict[var_id][target] = edge_idx_1, edge_idx_2

    # adding the edges between objects 
    def add_relations(self):

        self.rela_center = dict()

        # construct two centrual nodes to pass information
        for relation in CENTER_RELATION:
            rela_center_idx = len(self.nodes)
            self.rela_center[relation] = rela_center_idx
            self.add_node (rela_center_idx, NodeType.center_relation, relation)
            self.relations[relation] = []

        for relation in self.scene['relationships']:
            # We only care about right and behind, since the inverse is obvious
            if relation == "right" or relation == "behind":

                for object_id, related_objects in enumerate(self.scene["relationships"][relation]):
                    for related_object_id in related_objects:
                        
                        fst_idx = self.objs[object_id]
                        snd_idx = self.objs[related_object_id]
                        rela_idx = len(self.nodes)
                        center_idx = self.rela_center[relation]

                        self.add_node(rela_idx, NodeType.relation, relation)
                        self.add_undirected_edge(rela_idx, center_idx, EdgeType.center_relation)
                        self.add_undirected_edge(rela_idx, fst_idx, EdgeType.first)
                        self.add_undirected_edge(rela_idx, snd_idx, EdgeType.second)

                        self.relations[relation].append(rela_idx)

    def add_attr_or_rela(self):
        self.attr_or_rela = {}
        self.attr_or_rela.update(self.attrs)
        self.attr_or_rela.update(self.rela_center)

    def get_edge_info(self):
        valid_edges = list(filter(lambda e: not type(e) == type(None), self.edges))
        edge_infos = [edge.to_tuple() for edge in valid_edges]
        first, second, edge_type = zip(*edge_infos)
        return [ list(first), list(second)], list(edge_type)

    
    def update_binding(self, binding_dict):

        # # go through all the edges, remove all the bonding edge that of "bonding" type,
        # # but not in the bonding_dict
        # # (src_idx, dst_idx, edge_type, edge_attr, edge_embedding)
        # to_remove = []
        # for edge_idx, edge in enumerate(self.edges):
        #     if type(edge) == type(None):
        #         continue
        #     if edge.edge_type == EdgeType.bonding:
                
        #         # identify var name and obj names
        #         if "var" in self.nodes[edge.src_idx].name:
        #             var_name = self.nodes[edge.src_idx].name
        #             obj_name = self.nodes[edge.dst_idx].name
        #         elif "var" in self.nodes[edge.dst_idx].name:
        #             obj_name = self.nodes[edge.src_idx].name
        #             var_name = self.nodes[edge.dst_idx].name
        #         else:
        #             raise Exception("binding edge should have a var node")

        #         obj_name = obj_name.split('_')[1]

        #         # if the variable is not bond to anything, it is not mentioned in the
        #         # clauses, ignore the case
        #         if var_name not in bonding_dict.keys():
        #             continue
            
        #         # if the edge does not exists in the dict, delete the edge.
        #         if obj_name not in bonding_dict[var_name]:
        #             to_remove.append(edge_idx)

        for var_id, binding in self.vo_edge_dict.items():
            
            if not f"var_{var_id}" in binding_dict.keys():
                continue

            for target, (edge_idx_1, edge_idx_2) in binding.items():
                if not ((f"var_{var_id}", target) in binding_dict.items()):
                    self.remove_edge(edge_idx_1)
                    self.remove_edge(edge_idx_2)

def select_neighbor(node_idx, criteria):
    neighbors = graph.nodes[node_idx].neighbors
    selected_neighbors = set(filter(criteria, neighbors))
    return selected_neighbors

def get_obj_attr(graph, node_idx):
    assert (graph.nodes[node_idx].node_type == NodeType.obj)
    criteria = lambda neighbor_id: graph.nodes[neighbor_id].type == NodeType.attr
    return select_neighbor(node_idx, criteria)

def get_obj_rela(graph, node_idx):
    assert (graph.nodes[node_idx].node_type == NodeType.obj)
    criteria = lambda neighbor_id: graph.nodes[neighbor_id].type == NodeType.relation
    return select_neighbor(node_idx, criteria)

def get_obj_attr_or_rela(graph, node_idx):
    assert (graph.nodes[node_idx].node_type == NodeType.obj)
    criteria = lambda neighbor_id: (graph.nodes[neighbor_id].node_type == NodeType.relation) or (graph.nodes[neighbor_id].node_type == NodeType.attr)
    return select_neighbor(node_idx, criteria)

def get_bonded_var(graph, node_idx):
    assert (graph.nodes[node_idx].node_type == NodeType.obj)
    criteria = lambda neighbor_id: graph.nodes[neighbor_id].type == NodeType.var
    return select_neighbor(node_idx, criteria)

def get_bonded_obj(graph, node_idx):
    assert (graph.nodes[node_idx].node_type == NodeType.var)
    criteria = lambda neighbor_id: graph.nodes[neighbor_id].type == NodeType.obj
    return select_neighbor(node_idx, criteria)

# There are two sets of attributes we care about: 
#   1. what are the valid choices for the selected variables
#       - all the attributes that are related to the binded objects
#   2. What are the choices we want to aviod in these choices
#       - the common attributes shared accross every objects
def get_attr_mask(graph, var_node_id):
    assert (graph.nodes[var_id].node_type == NodeType.var)
    bonded_objs = get_bonded_obj(graph, var_node_id)
    # print(f"obj_neighbors: {obj_neighbors}"),......

    valid_attrs = set()
    common_attrs = set()

    for obj_ids in bonded_objs:
        obj_attrs = get_obj_attr(graph, obj_id)
        valid_attrs.update(obj_attrs)
        common_attrs.intersection_update(obj_attrs)

    # good candidate attributes
    candidate_attrs = valid_attrs.difference(common_attrs)
    return list(candidate_attrs)

# TODO: search the valid variable space
def search_valid_args(graph, rela_node_idx, var_node_idx):

    def get_first_arg(self, rela_node_idx, second_arg_node_idx): 
        assert (graph.nodes[rela_node_idx].node_type == NodeType.relation)
        assert (graph.nodes[second_arg_node_idx].node_type == NodeType.obj)
        rela_node = graph.nodes[rela_node_idx]
        assert (second_arg_node_idx in rela_node.neighbors)

        def criteria(neighbor_id):
            if not graph.nodes[neighbor_id].node_type == NodeType.obj:
                return False
            if (neighbor_id == second_arg_node_idx):
                return False
            
            first_arg_edge = rela_node.attached_edges[rela_node.neighbors.index(neighbor_id)]
            if first_arg_edge.edge_type == EdgeType.first:
                return False
            return True 
        return select_neighbor(node_idx, criteria)

    def get_second_arg(self, rela_node_idx, first_arg_node_idx): 
        assert (graph.nodes[rela_node_idx].node_type == NodeType.relation)
        assert (graph.nodes[first_arg_node_idx].node_type == NodeType.obj)
        rela_node = graph.nodes[rela_node_idx]
        assert (first_arg_node_idx in rela_node.neighbors)

        def criteria(neighbor_id):
            if not graph.nodes[neighbor_id].node_type == NodeType.obj:
                return False
            if (neighbor_id == first_arg_node_idx):
                return False
            
            second_arg_edge = rela_node.attached_edges[rela_node.neighbors.index(neighbor_id)]
            if second_arg_edge.edge_type == EdgeType.second:
                return False
            return True 
        return select_neighbor(node_idx, criteria)

    

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

if __name__ == "__main__":
    # We use a configuration file to specify the edges and nodes choices
    operation_list = ["left", "right", "front", "behind", "size", "color", "shape", "material"]
    choices = dict() 

    choices["size"] = ["large", "small"]
    choices["color"] = ["blue", "red", "yellow", "green", "gray", "brown", "purple", "cyan"]
    choices["shape"] = ["cube", "cylinder", "sphere"]
    choices["material"] = ["rubber", "metal"] 

    # edge_types = dict()
    # edge_types["attributes"] = ["size", "color", "shape", "material"]
    # edge_types["spatial_relation"] = ["left", "right", "front", "behind"]
    # edge_types["target"] = ["target"]
    # edge_types["bonding"] = ["bonding"]

    config = dict()
    config["operation_list"] = operation_list
    config["choices"] = choices
    # config["edge_types"] = edge_types

    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    graphs_path = os.path.join(raw_path, cmd_args.graph_file_name)
    scenes_path = os.path.abspath(os.path.join(raw_path, "unit_test_3.json"))

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