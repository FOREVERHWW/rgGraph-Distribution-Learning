import numpy as np
import torch
from torch_geometric.utils import to_dense_adj,dense_to_sparse,get_laplacian


device = 'cuda'
eps = 1e-7
def truncate(edge_indexF,mode,n):
    """
        mode:'label' or 'node'
        hetero_edge_index: edge_indexF
    """
    if mode =='l':
        f = edge_indexF[n:]
    else:
        f = edge_indexF[:n]
    return f

def updatehetero(edge_indexF,edge_indexl):
    """
    hetero_edge_index: edge_indexF
    updated label graph: edge_indexl
    goal: update hetero_edge_index with the updated label grpah
    """
    num_examples = edge_indexF.shape[0]-edge_indexl.shape[0]
    edge_indexF[num_examples:num_examples+edge_indexl.shape[0],num_examples:num_examples+edge_indexl.shape[1]] = edge_indexl
    return edge_indexF


def genhetero(edge_indexv,edge_indexl, Yv, train_mask):
    """
    node graph:edge_indexv 1711*1711 n*n matrix (adj)
    label graph:edge_indexl 4*4 m*m matrix (adj)
    node label: Yv
    hetero_edge_index = torch.zeros((n+m,n+m))

    """
    num_nodes = edge_indexv.size(dim=0)
    num_labels = edge_indexl.size(dim=1)
    hetero_edge_index = torch.zeros((num_nodes + num_labels, num_nodes + num_labels)).to(device)
    hetero_edge_index[:num_nodes, :num_nodes] = edge_indexv
    hetero_edge_index[num_nodes:, num_nodes:] = edge_indexl

    node_label_edge_index = torch.zeros((num_nodes, num_labels))
    for i in range(num_nodes):
        if train_mask[i]:
            val = torch.max(Yv[i])
            index = torch.argmax(Yv[i])
            node_label_edge_index[i][index] = val.item()
            # node_label_edge_index[i][index] = 1
    hetero_edge_index[:num_nodes, num_nodes:] = node_label_edge_index
    hetero_edge_index[num_nodes:, :num_nodes] = torch.t(node_label_edge_index)
    # for i in range(num_nodes):
    #     if train_mask[i]:
    #         val = torch.max(Yv[i])
    #         index = torch.argmax(Yv[i])
    #         hetero_edge_index[i][index+num_nodes] = val.item()
    #         hetero_edge_index[index+num_nodes][i] = val.item()
    return hetero_edge_index

def from_edge_index_to_adj(edge_index,edge_weight=None):
    adj = to_dense_adj(edge_index=edge_index,edge_attr=edge_weight)
    return adj[0]

def from_adj_to_edge_index(adj):
    edge_index = dense_to_sparse(adj)
    return edge_index[0],edge_index[1]
def genfeat(feature,y,train_mask):
    feature = feature[train_mask]
    y = y[train_mask]
    num_label = y.size(dim=1)
    num_example = y.size(dim=0)
    num_feature = feature.size(dim=1)
    # print(num_label)
    # print(num_example)
    # print(num_feature)
    feature_label = torch.zeros((num_label,num_feature)).to(device)
    feature_label = torch.add(feature_label,eps)
    count_label = np.zeros(num_label)
    for i in range(num_example):
        index = torch.argmax(y[i]).item()
        feature_label[index]+=feature[i]
        count_label[index]+=1
    for i in range(num_label):
        if count_label[i]!=0:
            feature_label[i]/=count_label[i]
        else:
            print("error on the dataset")
    # print(feature_label.shape)
    #print(feature_label[0])
    # print(count_label)
    return feature_label



def get_eigen(adj):
    eigenvalues = torch.linalg.eigvals(adj)
    # print(eigenvalues)
    eigen = eigenvalues.real
    sorted_eigen,_ = torch.sort(eigen)
    # print(sorted_eigen)
    return sorted_eigen
def clark(predict,actual):
    score = 0
    num = predict.shape[0]
    predict = predict.cpu().detach().numpy()
    actual = actual.cpu().detach().numpy()
    for i in range(num):
        score+=np.sqrt(np.sum(np.divide(np.square(actual[i]-predict[i]),np.square(actual[i]+predict[i]))))
    score = score/num
    return score
def intersection(predict,actual):
    score = 0
    num = predict.shape[0]
    predict = predict.cpu().detach().numpy()
    actual = actual.cpu().detach().numpy()
    for i in range(num):
        score+=np.sum(np.minimum(predict[i],actual[i]))
    score = score/num
    return score
def normalize(edge_index,edge_weight):
    lap = get_laplacian(edge_index,edge_weight=edge_weight,normalization='sym')
    # print(lap)
    return lap[0],lap[1]
def gcn_norm(adj):
    I = torch.eye(adj.shape[0]).to(device)
    adj = adj+I
    degrees = torch.sum(adj,1)
    degrees = torch.pow(degrees,-0.5)
    D = torch.diag(degrees)
    return torch.matmul(torch.matmul(D,adj),D)
def adj_norm(adj):
    D = torch.diag(torch.sum(adj,1))
    return D-adj

def homophily(edge_list,labels):
    print(edge_list)
    score = 0
    num_edges = len(edge_list[0])
    print(num_edges)
    for i in range(num_edges):
        if labels[edge_list[0][i]] == labels[edge_list[1][i]]:
            score+=1
    score = score/num_edges
    return score
