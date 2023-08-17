# deep learning packages
from torch_geometric.nn import GCNConv
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
import argparse
# visualization
import matplotlib.pyplot as plt
from utils import *
from mugcn import MUGCN
#from MUGCN5 import MUGCN
import csv

import builtins



# import torch_geometric.transforms as T
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--gcnhidden', type=int, default=32)
parser.add_argument('--pgehidden',type=int,default=32)
parser.add_argument('--decoderhidden',type=int,default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--alpha',type=float,default=0)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--gcnllayers',type=int,default=1)
parser.add_argument('--gcnvlayers',type=int,default=2)
parser.add_argument('--pgelayers',type=int,default=3)
parser.add_argument('--freqv',type=int,default=50)
parser.add_argument('--freql',type=int,default=100)
parser.add_argument('--patience',type=int,default=20)
parser.add_argument('--decoderlayers',type=int,default=3)
parser.add_argument('--mode',type=str,default='static',help='dynamic,static')
parser.add_argument('--normalize',action='store_true')
parser.add_argument('--inner_epochs',type=int,default=50)
parser.add_argument('--p',type=int,default=80)
args = parser.parse_args()

if args.gpu_id<0:
    device='cpu'
else:
    device= 'cuda'
    torch.cuda.manual_seed(args.seed)
    # torch.set_default_device('cuda')


# random seed setting
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
dataname = ''
# verbose
# print(args)
# Construct the homo Graph
data_path = './data/'
dataname = args.dataset
if args.dataset == 'dblp':
    data_file = np.load(data_path+dataname+"/APA_CC.npz")
    homo = geo.data.Data()
    homo.x = torch.tensor(data_file["x"])
    homo.y = torch.tensor(data_file["y"])
    homo.edge_index = torch.tensor(data_file["edge_index"])
elif args.dataset == 'yelp':
    data_file = np.load(data_path+dataname+"/bus_largest_cc.npz")
    homo = geo.data.Data()
    homo.x = torch.tensor(data_file["x"]).type(dtype=torch.float)
    homo.y = torch.tensor(data_file["y"])
    homo.edge_index = torch.tensor(data_file["edge_index"])
elif args.dataset =='acm':
    data_file = np.load(data_path+dataname+"/acm_largest_cc.npz")
    homo = geo.data.Data()
    homo.x = torch.tensor(data_file["x"]).type(dtype=torch.float)
    homo.y = torch.tensor(data_file["y"])
    homo.edge_index = torch.tensor(data_file["edge_index"])
elif args.dataset =='yelp2':
    data_file = np.load(data_path+dataname+"/yelp_bus_2.npz")
    homo = geo.data.Data()
    homo.x = torch.tensor(data_file["x"]).type(dtype=torch.float)
    homo.y = torch.tensor(data_file["y"])
    homo.edge_index = torch.tensor(data_file["edge_index"])

maskname = '/'
if args.p==50:
    maskname+='masks_50_20_30'
elif args.p == 40:
    maskname+='masks_40_30_30'
elif args.p == 80:
    maskname+='masks_80_10_10'
elif args.p ==30:
    maskname+='masks_30_30_40'
elif args.p==60:
    maskname+='masks_60_20_20'
else:
    print("error, no such masks")

model_state = './trained_model/'+dataname+'/'+mode+'/'+dataname+'50.pt'
adjfv_state = './trained_model/'+dataname+'/'+mode+'/'+dataname+'50_adjfv.npy'
adjv_state = './trained_model/'+dataname+'/'+mode+'/'+dataname+'50_adjv.npy'
fv_state = './trained_model/'+dataname+'/'+mode+'/'+dataname+'50_fv.npy'
for index_mask in range(1):
    BESTVAL = 100
    count = 1
    index_mask = 0
    masks = np.load(data_path+dataname+maskname+"/masks"+str(index_mask)+".npz")
    train_mask = torch.tensor(masks['train_mask'])
    val_mask = torch.tensor(masks['val_mask'])
    test_mask = torch.tensor(masks['test_mask'])
    mugcn = MUGCN(homo_dblp,train_mask,val_mask,test_mask,args,device)
    #best_val_loss,epoch,fv,adjfv,adjv,gcnv = mugcn.train()
    gcnv = torch.load(model_state)
    fv = torch.tensor(np.load(fv_state)).to(device)
    adjfv = torch.tensor(np.load(adjfv_state)).to(device)
    adjv = torch.tensor(np.load(adjv_state)).to(device)
    mugcn.load(fv,adjfv,adjv,gcnv)
    mugcn.test()
    #np.save('dblp_fv',fv.cpu())
    #np.save('dblp_adjfv',adjfv.cpu())
    #np.save('dblp_adjv',adjv.cpu())
# average_best_val_loss = 0
# average_result = np.zeros(6)  
# for index_mask in range(5):
#     args.freql = best_config[index_mask]['freql']
#     args.freqv = best_config[index_mask]['freqv']
#     args.pgehidden = best_config[index_mask]['pgehidden']
#     args.pgelayers = best_config[index_mask]['pgel']
#     args.gcnhidden = best_config[index_mask]['gcnhidden']
#     args.gcnllayers = best_config[index_mask]['gcnllayers']
#     args.gcnvlayers = best_config[index_mask]['gcnvlayers']
#     args.inner_epochs = best_config[index_mask]['inner_epochs']
#     args.weight_decay = best_config[index_mask]['weightdecay']
#     args.lr = best_config[index_mask]['lr']
#     masks = np.load("./"+dataname+"/masks"+str(index_mask)+".npz")
#     train_mask = torch.tensor(masks['train_mask'])
#     val_mask = torch.tensor(masks['val_mask'])
#     test_mask = torch.tensor(masks['test_mask'])
#     mugcn = MUGCN(homo_dblp,train_mask,val_mask,test_mask,args,device)
#     best_val_loss,epoch,fv,adjfv,adjv,gcnv = mugcn.train()
#     mugcn.load(fv,adjfv,adjv,gcnv)
#     performance = mugcn.test()
#     average_result+=performance
#     average_best_val_loss+=best_val_loss
# print("average val loss:",average_best_val_loss)
# print("average result:",average_result/5)
# print(best_val_loss)
# print(best_fv)
# print(best_adjfv)
# print(best_adjv)
# print(best_model)
# mugcn.load(best_fv,best_adjfv,best_adjv,best_model)
# unit test for utils
# debug purpose only
#feat = genfeat(homo_dblp.x,homo_dblp.y,train_mask)
# genhetero(edge_indexv,edge_indexl,Yv)
# adj = from_edge_index_to_adj(homo_dblp.edge_index).shape)
# adj_test = torch.zeros(7+num_label,7+num_label)
# updatehetero(adj_test,adj)
# edge_index,edge_weight = normalize(homo_dblp.edge_index,None)
# adj = from_edge_index_to_adj(edge_index,edge_weight)
# print(adj)

