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
from torch_geometric.nn import GraphConv, TopKPooling, NNConv, ARMAConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

from cmd_args import cmd_args
from scene2graph_prob import Graph, Edge, GraphNode
from utils import NodeType, EdgeType, Encoder
from functools import wraps 
from utils import get_config
from CGConv import CGConv
from SAGPooling import SAGPooling
    
class GNNGlobal(torch.nn.Module):
    def __init__(self, dataset, embedding_layer, hidden_dim = cmd_args.hidden_dim):
        super().__init__()

        self.embedding_layer = embedding_layer

        self.conv1 = CGConv(hidden_dim, dataset.num_edge_features)
        self.pool1 = SAGPooling(hidden_dim, min_score=0.0)
        self.conv2 = CGConv(hidden_dim, dataset.num_edge_features)
        self.pool2 = SAGPooling(hidden_dim, min_score=0.0)
        self.conv3 = CGConv(hidden_dim, dataset.num_edge_features)
        self.pool3 = SAGPooling(hidden_dim, min_score=0.0)
        self.conv4 = CGConv(hidden_dim, dataset.num_edge_features)
        self.pool4 = SAGPooling(hidden_dim, min_score=0.0)

        self.lin1 = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.embedding_layer(x)

        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv4(x, edge_index, edge_attr=edge_attr))
        x, edge_index, edge_attr, batch, _, _ = self.pool4(x, edge_index, edge_attr, batch)
        x4 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3 + x4

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

