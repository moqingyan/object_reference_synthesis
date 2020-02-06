import sys
import os 
import json 
import pickle 
from cmd_args import cmd_args
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


class Graph2(object):
    
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
        self.vo_node_dict = {}
        self.init_vo_edge_dict()

        self.vars = {}
        self.add_variables()

        if cmd_args.global_node:
            self.add_global_node()

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
            self.vo_node_dict[var_id] = {}

    def add_node(self, index, node_type, name): 
        self.nodes.append(GraphNode(index, node_type, name)) 
    
    def remove_node(self, index):
        self.nodes.remove(index)

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
            # add a node between binding
            for target in range(self.obj_num):
                target_idx = self.objs[target]
                binding_node_idx = len(self.nodes)
                # self.add_node(binding_node_idx, NodeType.binding, f"binding_{binding_node_idx}")
                
                edge_idx_1, edge_idx_2 = self.add_undirected_edge(var_idx, target_idx, EdgeType.bonding)
                # edge_idx_3, edge_idx_4 = self.add_undirected_edge(binding_node_idx, target_idx, EdgeType.bonding)

                self.vo_edge_dict[var_id][target] = edge_idx_1, edge_idx_2
                self.vo_node_dict[var_id][target] = binding_node_idx

    # one way global node
    def add_global_node(self):
        self.global_idx = len(self.nodes)
        self.add_node(self.global_idx, NodeType.gb, "gb")
        for node_id in range(len(self.nodes)):
            self.add_edge(node_id, self.global_idx, EdgeType.gb)
            
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

    def get_nodes(self):
        return [ node.name for node in graph.nodes ]
    
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

    def get_attr_locs(self):
        return self.attrs.values()

    def get_rela_locs(self):
        return self.rela_center.values()

    def get_attr_or_rela_locs(self):
        return self.attr_or_rela.values()

    def get_var_locs(self, rel=None):
        if not (type(rel) == type(None)):
            locs = []
            for r in rel:
                locs.append(self.vars[r])
            return locs
        else:
            return list(self.vars.values())

    def get_attr_by_idx(self, idx):
        return list(self.attrs.keys())[sel]

    def get_var_by_idx(self, idx):
        return list(self.vars.keys())[sel]

    def get_rela_by_idx(self, idx):
        return list(self.rela_center.keys())[sel]

    def get_attr_or_rela_by_idx(self, idx):
        return list(self.attr_or_rela.keys())[sel]

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


class Graph:

    def __init__(self, config, scene, target_id, max_var_num = cmd_args.max_var_num, hidden_dim = cmd_args.hidden_dim):
        self.config = config 
        self.scene = scene
        self.target_id = target_id
        self.obj_num = len(self.scene['objects'])
        self.max_var_num = max_var_num

        self.base_graph = BaseDiGraph() 
        self.setup_graph_dict()

    def setup_graph_dict(self):

        self.base_graph.add_vertex("target")
        
        # Construct attributes as nodes
        self.attrs = []
        for attrs in self.config["choices"].values():
            for attr in attrs:
                self.attrs.append(attr)
                self.base_graph.add_vertex(attr)

        attr_count = 0
        # Construct object as object nodes
        for obj_id, obj in enumerate(self.scene['objects']):
            self.base_graph.add_vertex(f"obj_{obj_id}")
            
            if cmd_args.attr_inter_node:
                # add in edges between the object and its attributes
                for key in self.config["choices"].keys():
                    obj_attr = obj[key]
                    self.base_graph.add_vertex(f"attr_{attr_count}")
                    self.base_graph.add_undirected_edge(f"obj_{obj_id}", f"attr_{attr_count}", EdgeType.edge_attr)
                    self.base_graph.add_undirected_edge(f"attr_{attr_count}", obj_attr, EdgeType.edge_attr)
                    attr_count += 1
            
            else:
                # add in edges between the object and its attributes
                for key in self.config["choices"].keys():
                    obj_attr = obj[key]
                    self.base_graph.add_undirected_edge(f"obj_{obj_id}", obj_attr, EdgeType.edge_attr)
                    attr_count += 1
                

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
    clauses = get_all_clauses(config)

    data_dir = os.path.abspath(__file__ + "../../../../data")
    raw_path = os.path.abspath(os.path.join(data_dir, "./processed_dataset/raw"))
    graphs_path = os.path.join(raw_path, "unit_test_3_graphs.pkl")
    scenes_path = os.path.abspath(os.path.join(raw_path, "unit_test_3.json"))

    # In the pytorch geometry package, only int and tensor seems to be allowed to save
    # we process all the graphs and save them to a file.
    with open(scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)
    
    graphs = []

    for scene in scenes:
        for target_id in range(len(scene["objects"])):
            graph = Graph(config, scene, target_id)
            # unary_clauses_idx, binary_clauses_idx = state.get_clauses_idx()
            # clauses_idx = unary_clauses_idx + binary_clauses_idx 
            # print(clauses_idx)
            graphs.append(graph)
    
    edge_info = graph.get_edge_info()
    with open(graphs_path, 'wb') as graphs_file:
        pickle.dump(graphs, graphs_file) 
    
    config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'w') as config_file:
        json.dump(config, config_file)