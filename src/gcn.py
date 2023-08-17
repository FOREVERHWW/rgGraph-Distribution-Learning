# deep learning packages
from layers import GCNConv
import torch
import torch_geometric as geo
from torch.nn import Linear
import torch.nn.functional as F
import sklearn
from sklearn.metrics import mean_squared_error

# Main computation libraries
import scipy.sparse as sp
from scipy.spatial import distance
import numpy as np
import scipy as sp
import math
import random
import networkx as nx

# visualization
import matplotlib.pyplot as plt


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels, layers=2):
        """
        initializes a GCN with two layers
        :param input_dim: the dimension of your input
        :param input_dim: the dimension of your output
        :param hidden_channels: the amount of channels in the layers that aren't input or output
        :param layers: the number of layers you want in your model
        """
        super().__init__()
        # seed used so this model trains the same way everytime
        self.layers = []
        for i in range(layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_channels))
            else:
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(torch.nn.Linear(hidden_channels,input_dim))
        self.layers.append(torch.nn.Linear(input_dim,hidden_channels))
        self.layers.append(GCNConv(hidden_channels,output_dim))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x, adj,adj2,mode):
        """
        Used to progress the model
        :param x: feature matrix
        :param edge_index: edge matrix
        :return:
        """
        for i, conv in enumerate(self.layers):
            if i <len(self.layers)-3:
                if i == 0:
                    x = conv(x,adj)
                else:
                    x = conv(x,adj2)
            elif (i==len(self.layers)-3) or (i == len(self.layers)-2):
                x = conv(x)
            else:
                x = conv(x,adj2)
            if i != len(self.layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=0, training=self.training)
        if mode =='l':
            return x
        else:
            return F.softmax(x, dim=1)
    @torch.no_grad()
    def get_embedding(self,x,adj,adj2):
        self.eval()
        for i, conv in enumerate(self.layers):
            if i == 0:
                x = conv(x,adj)
            else:
                if i < len(self.layers)-2:
                    if i != len(self.layers)-3:
                        x = conv(x, adj2)
                    else:
                        x = conv(x)
            #if i !=len(self.layers)-3:
                #x = x.relu()
            x = x.relu()
            # x = F.dropout(x, p=0.5, training=self.training)
        return x
    def initialization():
        pass
    def reset_parameters(self):
        def weight_reset(m):
            if isinstance(m, nn.GCNConv):
                m.reset_parameters()
        self.apply(weight_reset)
