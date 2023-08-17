from run_model import run_model
import torch
import torch_geometric as geo
import numpy as np
import argparse
import random
seed = 12345
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--p', type=int, default='50')
parser.add_argument('--start_mask', type=int, default=0)
parser.add_argument('--auto_stop', type=bool, default=True)
parser.add_argument('--add', type=bool, default=False)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--lr', type=float, default=.0001)
parser.add_argument('--weight_decay', type=float, default=.0005)
parser.add_argument('--max_epochs', type=int, default=500)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--display', type=bool, default=False)

args = parser.parse_args()

dataname = ''
# verbose
# print(args)
# Construct the homo Graph
data_path = './data/'
dataname = args.dataset
if args.dataset == 'dblp':
    model_name = 'dblp50_gcn.pt'
    data_file = np.load(data_path+dataname+"/APA_CC.npz")
    homo = geo.data.Data()
    homo.x = torch.tensor(data_file["x"])
    homo.y = torch.tensor(data_file["y"])
    homo.edge_index = torch.tensor(data_file["edge_index"])
elif args.dataset == 'yelp':
    model_name = 'yelp50_gcn.pt'
    data_file = np.load(data_path+dataname+"/bus_largest_cc.npz")
    homo = geo.data.Data()
    homo.x = torch.tensor(data_file["x"]).type(dtype=torch.float)
    homo.y = torch.tensor(data_file["y"])
    homo.edge_index = torch.tensor(data_file["edge_index"])
elif args.dataset =='acm':
    model_name = 'acm50_gcn.pt'
    data_file = np.load(data_path+dataname+"/acm_largest_cc.npz")
    homo = geo.data.Data()
    homo.x = torch.tensor(data_file["x"]).type(dtype=torch.float)
    homo.y = torch.tensor(data_file["y"])
    homo.edge_index = torch.tensor(data_file["edge_index"])
elif args.dataset =='yelp2':
    model_name = 'yelp2_50_gcn.pt'
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
index_mask = 0
masks = np.load(data_path+dataname+maskname+"/masks"+str(index_mask)+".npz")
homo.test_mask = torch.tensor(masks["test_mask"])
homo.val_mask = torch.tensor(masks["val_mask"])
homo.train_mask = torch.tensor(masks["train_mask"])
homo.true_perc = torch.tensor(masks["true_perc"])

avg_val_loss = 0
avg_epochs = 0
avg_eval_dict = {"cheb": 0, "cos": 0, "canb": 0, "clark": 0, "int": 0, "kl": 0}
model_path = './trained_model/'+dataname+'/'+model_name
for i in range(1):
    val_loss, epochs, eval_dict = run_model(homo, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                                            lr=args.lr, weight_decay=args.weight_decay, display=args.display,
                                            num_epochs=args.epochs, patience=args.patience,
                                            auto_stop=args.auto_stop, max_epoch=args.max_epochs,model_path=model_path)
    avg_val_loss += val_loss
    avg_epochs += epochs
    for k, v in eval_dict.items():
        avg_eval_dict[k] += v
avg_val_loss /= 1
avg_epochs /= 1
for k, v in avg_eval_dict.items():
    avg_eval_dict[k] = v / 1

print(avg_val_loss, avg_epochs, avg_eval_dict)


