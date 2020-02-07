import os
import os.path as osp
import sys
import json
import pickle

import torch
import torch.nn.functional as F
import torch.nn as nn

from tqdm import tqdm
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, NNConv, SAGPooling, ARMAConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from cmd_args import cmd_args
from scene2graph import Graph, Edge, GraphNode
from utils import NodeType, EdgeType, Encoder
from functools import wraps 
from utils import get_config
from graphConvE import GraphConvE

def add_method(cls):
    def decorator(func):
        @wraps(func) 
        def wrapper(self, *args, **kwargs): 
            return func(self, *args, **kwargs)
        setattr(cls, func.__name__, wrapper)
        # Note we are not binding func, but wrapper which accepts self but does exactly the same as func
        return func # returning func means func can still be used normally
    return decorator


def graph2data(graph, attr_encoder):
    x = attr_encoder.get_embedding(graph.get_nodes())
    edge_index, edge_types = graph.get_edge_info()
    edge_attrs = torch.tensor(attr_encoder.get_embedding([f"edge_{tp}" for tp in edge_types]))
    return Data(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attrs), graph.target_id)
    
# @add_method(Data)
# def update_data(self, graph, attr_encoder):
#     x = attr_encoder.get_embedding( [ node.name for node in graph.nodes])
#     edge_index, edge_types = graph.get_edge_info()
#     edge_attrs = torch.tensor(attr_encoder.get_embedding(edge_types))
#     self.edge_attrs = edge_attrs
#     self.edge_index = torch.tensor(edge_index)
#     self.x = torch.tensor(x)

class SceneDataset(InMemoryDataset):
    def __init__(self, root, config, transform=None, pre_transform=None):
        self.config = config
        self.attr_encoder = Encoder(config)

        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        # return ["graphs.pkl"]
        return [cmd_args.graph_file_name]

    @property
    def processed_file_names(self):
        # return ['train_1000_dataset.pt']
        return [cmd_args.dataset_name]

    def download(self):
        pass

    def process(self):
        data_list = []
        
        for raw_path in self.raw_paths:
            with open (raw_path, 'rb') as raw_file:
                graphs = pickle.load(raw_file)

            for graph_id, graph in enumerate(graphs):

                x = self.attr_encoder.get_embedding(graph.get_nodes())
                edge_index, edge_types = graph.get_edge_info()
                edge_attrs = self.attr_encoder.get_embedding([f"edge_{tp}" for tp in edge_types])

                # edge_attrs = torch.tensor(self.attr_encoder.get_embedding(edge_types))
                data_point = Data(torch.tensor(x), torch.tensor(edge_index), torch.tensor(edge_attrs), graph.target_id)
                
                # print(torch.tensor(x), torch.tensor(edge_index), edge_attrs, graph.target_id)
                data_point.obj_num = len(graph.scene["objects"])
                data_point.graph_id = graph_id
                # data_point.attr_encoder = self.attr_encoder
                data_list.append(data_point)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def create_dataset(data_dir, scenes_path, graphs_path, config=None):
    with open(scenes_path, 'r') as scenes_file:
        scenes = json.load(scenes_file)
        if type(config) == type(None):
            config = get_config()
        graphs = []

        for scene in scenes:
            for target_id in range(len(scene["objects"])):
                graph = Graph(config, scene, target_id, ground_truth_scene=scene["ground_truth"])
                graphs.append(graph)

        with open(graphs_path, 'wb') as graphs_file:
            pickle.dump(graphs, graphs_file) 

        root = os.path.join(data_dir, "./processed_dataset")
        scene_dataset = SceneDataset(root, config)
    return graphs, scene_dataset

class GNNLocal(torch.nn.Module):
    def __init__(self, dataset, embedding_layer, hidden_dim = cmd_args.hidden_dim):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.conv = ARMAConv( hidden_dim, hidden_dim, num_layers=4)

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, int(hidden_dim/2))
        self.lin3 = torch.nn.Linear(int(hidden_dim/2), cmd_args.embedding_dim)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding_layer(x)
        edge_attr = edge_attr.float() / edge_attr.max()
        # edge_attr = self.embedding_layer(edge_attr)

        x = F.relu(self.conv(x, edge_index, edge_weight=edge_attr))
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin2(x))
        x = torch.tanh(self.lin3(x))
        # print (x.shape)
        return x
    
class GNNGlobal(torch.nn.Module):
    def __init__(self, dataset, embedding_layer, hidden_dim = cmd_args.hidden_dim):
        super().__init__()

        self.embedding_layer = embedding_layer

        self.conv1 = GraphConvE(hidden_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim)
        self.conv2 = GraphConvE(hidden_dim, hidden_dim,)
        self.pool2 = SAGPooling(hidden_dim)
        self.conv3 = GraphConvE(hidden_dim, hidden_dim)
        self.pool3 = SAGPooling(hidden_dim)
        self.conv4 = GraphConvE(hidden_dim, hidden_dim)
        self.pool4 = SAGPooling(hidden_dim)

        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding_layer(x)
        edge_attr = self.embedding_layer(edge_attr)

        x = F.relu(self.conv1(x, edge_index, edge_weight=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_weight=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_weight=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index, edge_weight=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

class GNNGL(torch.nn.Module):
    def __init__(self, dataset, embedding_layer, hidden_dim = cmd_args.hidden_dim):
        super().__init__()
        if not cmd_args.global_node:
            self.global_layer = GNNGlobal( dataset, embedding_layer, hidden_dim)
        self.local_layer = GNNLocal( dataset, embedding_layer, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.embedding_layer = self.local_layer.embedding_layer

    def forward(self, data):
        local_embedding = self.local_layer(data)
        global_embedding = None
        if not cmd_args.global_node:
            global_embedding = self.global_layer(data)
        
        return local_embedding, global_embedding

class GNN(torch.nn.Module):
    def __init__(self, dataset, embedding_layer, hidden_dim = cmd_args.hidden_dim):
        super().__init__()

        self.embedding_layer = embedding_layer

        self.conv1 = GraphConvE(hidden_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim)
        self.conv2 = GraphConvE(hidden_dim, hidden_dim,)
        self.pool2 = SAGPooling(hidden_dim)
        self.conv3 = GraphConvE(hidden_dim, hidden_dim)
        self.pool3 = SAGPooling(hidden_dim)
        self.conv4 = GraphConvE(hidden_dim, hidden_dim)
        self.pool4 = SAGPooling(hidden_dim)

        self.lin1 = torch.nn.Linear(hidden_dim*3, hidden_dim*2)
        self.lin2 = torch.nn.Linear(hidden_dim*2, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.l1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding_layer(x)
        # edge_attr = self.embedding_layer(edge_attr)

        x = F.relu(self.conv1(x, edge_index))
        x_local = self.l1(x)
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index, edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

       # x = x1 + x2 + x3 + x4
        x_global = (x1 + x2 + x3 + x4) 
        x = torch.cat((x_local, x_global.repeat(x_local.shape[0], 1)), dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)
        # print (x.shape)
        return x

class GNNNode(torch.nn.Module):
    def __init__(self, dataset, embedding_layer, hidden_dim = cmd_args.hidden_dim):
        super().__init__()

        self.embedding_layer = embedding_layer
        self.nn2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim**2))
        self.nn4 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), 
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim**2))

        self.conv1 = GraphConvE(hidden_dim, hidden_dim)
        self.pool1 = TopKPooling(hidden_dim, ratio=1.0)
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.nn2)
        self.pool2 = TopKPooling(hidden_dim, ratio=1.0)
        self.conv3 = GraphConvE(hidden_dim, hidden_dim)
        self.pool3 = TopKPooling(hidden_dim, ratio=1.0)
        self.conv4 = NNConv(hidden_dim, hidden_dim, self.nn4)
        self.pool4 = TopKPooling(hidden_dim, ratio=1.0)

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, hidden_dim)

        self.l1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.l4 = torch.nn.Linear(hidden_dim, hidden_dim)


    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding_layer(x)
        x = F.relu(self.conv1(x, edge_index))
        
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.3, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

if __name__ == "__main__":
    data_dir = os.path.abspath(__file__ + "../../../../data")
    root = os.path.join(data_dir, "./processed_dataset")
    config_path = config_path = os.path.join(data_dir, "config.json")
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
    scene_dataset = SceneDataset(root, config)
    scene_dataset.shuffle()
    print("blah")

    # def test(loader):
    #     model.eval()

    #     correct = 0
    #     for data in loader:
    #         data = data.to(device)
    #         pred = model(data).max(dim=1)[1]
    #         correct += pred.eq(data.y).sum().item()
    #     return correct / len(loader.dataset)


    # for epoch in range(1, 201):
    #     loss = train(epoch)
    #     train_acc = test(train_loader)
    #     test_acc = test(test_loader)
    #     print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
    #         format(epoch, loss, train_acc, test_acc))
