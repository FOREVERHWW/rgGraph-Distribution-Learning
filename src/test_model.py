from run_functions import run_model
import torch
import torch_geometric as geo
import numpy as np
import argparse

torch.cuda.manual_seed(12345)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--mask_split', type=str, default='80_10_10')
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
parser.add_argument('--num_runs', type=int, default=1)

args = parser.parse_args()
data_path = "data/"
if args.dataset == "yelp":
    data_path += "yelp/bus_largest_cc"
elif args.dataset == "yelp2":
    data_path += "yelp2/yelp_bus_2"
elif args.dataset == "dblp":
    data_path += "dblp/APA_CC"
elif args.dataset == "acm":
    data_path += "acm/acm_sub_subgraph"
elif args.dataset == "acm_condensed":
    data_path += "acm_condensed/acm_condensed_label_subgraph"
elif args.dataset == "imdb":
    data_path += "imdb/imdb_cc"
data_path += ".npz"

file_name = "data/" + args.dataset + "/results/hyperparameter_search_" + args.mask_split
mask_path = "data/" + args.dataset + "/masks_" + args.mask_split

data_file = np.load(data_path)
data = geo.data.Data()
data.x = torch.tensor(data_file["x"], dtype=torch.float)
data.y = torch.tensor(data_file["y"])
data.edge_index = torch.tensor(data_file["edge_index"])

masks = np.load(mask_path + "/masks0.npz")
data.test_mask = torch.tensor(masks["test_mask"])
data.val_mask = torch.tensor(masks["val_mask"])
data.train_mask = torch.tensor(masks["train_mask"])
data.true_perc = torch.tensor(masks["true_perc"])


avg_val_loss = 0
avg_epochs = 0
avg_eval_dict = {"cheb": 0, "cos": 0, "canb": 0, "clark": 0, "int": 0, "kl": 0}

for i in range(args.num_runs):
    val_loss, epochs, eval_dict = run_model(data, hidden_channels=args.hidden_channels, num_layers=args.num_layers,
                                            lr=args.lr, weight_decay=args.weight_decay, display=args.display,
                                            num_epochs=args.epochs, patience=args.patience,
                                            auto_stop=args.auto_stop, max_epoch=args.max_epochs, save_model=True,
                                            data_name=args.dataset + args.mask_split)
    avg_val_loss += val_loss
    avg_epochs += epochs
    for k, v in eval_dict.items():
        avg_eval_dict[k] += v
avg_val_loss /= args.num_runs
avg_epochs /= args.num_runs
for k, v in avg_eval_dict.items():
    avg_eval_dict[k] = v / args.num_runs

print(avg_val_loss, avg_epochs, avg_eval_dict)


