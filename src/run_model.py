from DBLP_Model import GCN, train, test, calc_val_loss, chebyshev, canberra, clark, intersection, entropy
from scipy.spatial.distance import cosine
import numpy as np
import scipy as sp
import csv
import argparse
import torch
import torch_geometric as geo
import networkx as nx

# visualization
import matplotlib.pyplot as plt
#from preprocess_DBLP import create_masks

# # Construct the Homo_DBLP Graph
# homo_dblp_file = np.load("data/dblp/APA_CC.npz")
# homo_dblp = geo.data.Data()
# homo_dblp.x = torch.tensor(homo_dblp_file["x"])
# homo_dblp.y = torch.tensor(homo_dblp_file["y"])
# homo_dblp.edge_index = torch.tensor(homo_dblp_file["edge_index"])
#
# # create_masks(homo_dblp, .8, .1, .1, save=True, random_state=123456)
# masks = np.load("data/dblp/masks_80_10_10/masks0.npz")
# test_mask = torch.tensor(masks["test_mask"])
# val_mask = torch.tensor(masks["val_mask"])
# train_mask = torch.tensor(masks["train_mask"])
# true_perc = masks["true_perc"]
#
# homo_dblp.train_mask = train_mask
# homo_dblp.test_mask = test_mask
# homo_dblp.val_mask = val_mask
# homo_dblp.true_perc = true_perc


# make random edge graph

# num_nodes = homo_dblp.x.size(dim=0)
# rand_graph = nx.fast_gnp_random_graph(num_nodes, homo_dblp.edge_index.size(dim=1)/(num_nodes ** 2), seed=134861, directed=False)
# adj = nx.adjacency_matrix(rand_graph)
# rand_edges = nx.to_edgelist(rand_graph)
# edge_tens = [[], []]
# for edge in rand_edges:
#     edge_tens[0].append(edge[0])
#     edge_tens[1].append(edge[1])
#
# rand_graph = geo.data.Data(x=homo_dblp.x, y= homo_dblp.y, edge_index=torch.tensor(edge_tens))
# rand_graph.train_mask = train_mask
# rand_graph.test_mask = test_mask


def run_model(model_data: geo.data.Data, hidden_channels=64, num_layers=3, lr=.001, weight_decay=5e-4, num_epochs=100,
              display=True, auto_stop=False, conv_crit=1e-4, patience=10, max_epoch = 500,model_path=None):
    """
    Use this function to run the model
    :param model_data: the graph data you want to run the model on
    :param hidden_channels: number of hidden channels
    :param num_layers: number of layers in the model
    :param lr: learning rate of the optimizer
    :param weight_decay: weight decay of the optimizer
    :param num_epochs: the number of epochs to run
    :param display: whether to graph the evaluation metrics or not
    :param auto_stop: whether to autostop or use epochs
    :param conv_crit: how close two consecutive validation losses must be to be considered converged
    :param patience: how many times the convergence criteria must be broken before stopping
    :param max_epoch: maximum epochs to run during autostop
    :return: The minimum validation loss, the number of epochs (if autostopped),
             a dictionary that contains all the final evaluation metrics
    """
    output_dimension = model_data.y.size(dim=1)
    # output is has hidden_channels number of latent features
    model = GCN(model_data.num_features, output_dimension, hidden_channels=hidden_channels, layers=num_layers, lr=lr,
                weight_decay=weight_decay)
    # send model to gpu if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.load_state_dict(model_path)
    model_data.to(device)

    # graph evaluation metrics
    loss_arr = []
    val_arr = []
    cheb_arr = []
    cos_arr = []
    canb_arr = []
    clark_arr = []
    int_arr = []
    kl_arr = []

    autostop_epoch = 1
    min_val_loss = 1e3
    if auto_stop:
        patience_count = 0
        last_loss = 1e3
        while patience_count <= patience and autostop_epoch < max_epoch:
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            loss = train(model, model_data, kl_loss)
            val_loss = calc_val_loss(model, model_data, kl_loss)
            if display:
                val_mask = model_data.val_mask
                cheb_eval = test(model, model_data, val_mask, eval_func=chebyshev)
                cos_eval = test(model, model_data, val_mask, eval_func=cosine)
                canb_eval = test(model, model_data, val_mask, eval_func=canberra)
                clark_eval = test(model, model_data, val_mask, eval_func=clark)
                int_eval = test(model, model_data, val_mask, eval_func=intersection)
                kl_eval = test(model, model_data, val_mask, eval_func=entropy)

                loss_arr.append(loss.item())
                val_arr.append(val_loss.item())
                cheb_arr.append(cheb_eval)
                cos_arr.append(cos_eval)
                canb_arr.append(canb_eval)
                clark_arr.append(clark_eval)
                int_arr.append(int_eval)
                kl_arr.append(kl_eval)

            if min_val_loss < val_loss:
                patience_count += 1
            else:
                if abs(last_loss - val_loss) < conv_crit:
                    patience_count += 1
                else:
                    patience_count = 0
                min_val_loss = val_loss

            if display:
                print(
                    f'Epoch: {autostop_epoch:03d}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Chebyshev Distance: {cheb_eval:.4f}'
                    f', Cosine Distance: {cos_eval:.4f}, Canberra Distance: {canb_eval:.4f}, Clark Distance: {clark_eval:.4f}'
                    f', Intersection Distance: {int_eval:.4f}, KL Divergence Distance: {kl_eval:.4f}')
            autostop_epoch += 1
    else:
        for epoch in range(1, num_epochs + 1):
            # MSE = torch.nn.MSELoss(reduction='sum')
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            loss = train(model, model_data, kl_loss)
            val_loss = calc_val_loss(model, model_data, kl_loss)
            val_mask = model_data.val_mask
            min_val_loss = min(min_val_loss, val_loss)

            if display:
                cheb_eval = test(model, model_data, val_mask, eval_func=chebyshev)
                cos_eval = test(model, model_data, val_mask, eval_func=cosine)
                canb_eval = test(model, model_data, val_mask, eval_func=canberra)
                clark_eval = test(model, model_data, val_mask, eval_func=clark)
                int_eval = test(model, model_data, val_mask, eval_func=intersection)
                kl_eval = test(model, model_data, val_mask, eval_func=entropy)
                loss_arr.append(loss.item())
                val_arr.append(val_loss.item())
                cheb_arr.append(cheb_eval)
                cos_arr.append(cos_eval)
                canb_arr.append(canb_eval)
                clark_arr.append(clark_eval)
                int_arr.append(int_eval)
                kl_arr.append(kl_eval)
                print(
                    f'Epoch: {epoch:03d}, Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, Chebyshev Distance: {cheb_eval:.4f}'
                    f', Cosine Distance: {cos_eval:.4f}, Canberra Distance: {canb_eval:.4f}, Clark Distance: {clark_eval:.4f}'
                    f', Intersection Distance: {int_eval:.4f}, KL Divergence Distance: {kl_eval:.4f}')
    torch.save(model.state_dict(),'yelp50_gcn.pt')
    test_mask = model_data.test_mask
    cheb_eval = test(model, model_data, test_mask, eval_func=chebyshev)
    cos_eval = test(model, model_data, test_mask, eval_func=cosine)
    canb_eval = test(model, model_data, test_mask, eval_func=canberra)
    clark_eval = test(model, model_data, test_mask, eval_func=clark)
    int_eval = test(model, model_data, test_mask, eval_func=intersection)
    kl_eval = test(model, model_data, test_mask, eval_func=entropy)
    print(f'Test Evaluation - Loss: {loss.item():.4f}, Chebyshev Distance: {cheb_eval:.4f}'
          f', Cosine Distance: {cos_eval:.4f}, Canberra Distance: {canb_eval:.4f}, Clark Distance: {clark_eval:.4f}'
          f', Intersection Distance: {int_eval:.4f}, KL Divergence Distance: {kl_eval:.4f}')

    eval_dict = {"cheb": cheb_eval, "cos": cos_eval, "canb": canb_eval, "clark": clark_eval, "int": int_eval,
                 "kl": kl_eval}

    # Visualize the evaluation metrics
    if display:
        epochs = np.array(range(1, len(loss_arr) + 1))
        loss_arr = np.array(loss_arr)
        cheb_arr = np.array(cheb_arr)
        cos_arr = np.array(cos_arr)
        canb_arr = np.array(canb_arr)
        clark_arr = np.array(clark_arr)
        int_arr = np.array(int_arr)
        kl_arr = np.array(kl_arr)
        fig, ax = plt.subplots(7)
        fig.set_figwidth(10)
        fig.set_figheight(14)
        fig.suptitle(f'KL Div Loss Model, Hidden Channels: {hidden_channels}, Layers: {num_layers}, '
                     f'Training Percentage: {model_data.true_perc[0]:.4f}\n'
                     f'Validation Percentage: {model_data.true_perc[1]:.4f}, Test Percentage: {model_data.true_perc[2]:.4f}'
                     f', Learning Rate: {lr}, Weight Decay: {weight_decay}')
        ax[0].plot(epochs, loss_arr, label="Training Loss")
        ax[0].plot(epochs, val_arr, label="Validation Loss")
        ax[0].set(ylabel='KL-Divergence')
        ax[0].legend()
        ax[1].plot(epochs, cos_arr, label="Cosine")
        ax[1].legend()
        ax[1].set(ylabel='Average Distance')
        ax[2].plot(epochs, canb_arr, label="Canberra")
        ax[2].set(ylabel='Average Distance')
        ax[2].legend()
        ax[3].plot(epochs, clark_arr, label="Clark")
        ax[3].set(ylabel='Average Distance')
        ax[3].legend()
        ax[4].plot(epochs, kl_arr, label="KL Divergence")
        ax[4].set(ylabel='Average Distance')
        ax[4].legend()
        ax[5].plot(epochs, cheb_arr, label="Chebyshev")
        ax[5].set(ylabel='Average Distance')
        ax[5].legend()
        ax[6].plot(epochs, int_arr, label="Intersection")
        ax[6].set(xlabel='Epochs', ylabel='Average Distance')
        ax[6].legend()
        plt.show()

    if auto_stop:
        return min_val_loss.item(), autostop_epoch, eval_dict
    else:
        return min_val_loss.item(), num_epochs, eval_dict


def optimize_hyperparameters(data_file_name: str, hidden_channels, num_layers, lr, weight_decay, num_epochs=None,
                             start_mask=0, num_runs=5, file_name="data/optimize_hyperparameters.csv", add=False, display=False,
                             auto_stop=False, conv_crit=1e-3, patience=10, mask_path="data"):
    """
    finds the optimal inputted combination of hyperparameters based on validation loss
    :param data_file_name: a string representing the path to the dataset you want to optimize the hyperparameters to
    :param hidden_channels: a list of number of hidden channels
    :param num_layers: a list of number of layers
    :param lr: a list of learning rates
    :param weight_decay: a list of weight decay rates
    :param num_epochs: a list of number of epochs
    :param start_mask: the mask you want to start on
    :param num_runs: number of runs you want to average over
    :param file_name: a string representing the name of the file to save the data to
    :param add: whether you are appending to a csv file or not
    :param display: whether you want to graph the model's results or not
    :param auto_stop: whether you want the model to autostop or for it to run a certain number of epochs
    :param conv_crit: how close two consecutive validation losses must be to be considered converged
    :param patience: how many times the convergence criteria must be broken before stopping
    :param mask_path: path to the directory that holds the masks we want to use for the experiment
    :return:
    """

    if num_epochs is None:
        num_epochs = [100]

    # ensure file name has .csv extension
    if file_name.find(".") == -1:
        file_name += ".csv"

    file = open(file_name, 'a', newline='')
    writer = csv.writer(file)
    # add headers
    if not add:
        field = ["Mask", "Hidden Channels", "Number of Layers", "Learning Rate", "Weight Decay", "Number of Epochs",
                 "Validation Loss", "Chebyshev Distance", "Cosine Distance", "Canberra Distance",
             "Clark Distance", "Intersection Distance", "KL Divergence Distance"]
        writer.writerow(field)
    tot_comb = len(hidden_channels) * len(num_layers) * len(lr) * len(weight_decay) * len(num_epochs) * num_runs
    curr_comb = 1

    best_hyperparams = [tuple()] * num_runs
    best_val = [1e3] * num_runs
    best_results = [tuple()] * num_runs
    # Construct the data graph
    data_file = np.load(data_file_name)
    data = geo.data.Data()
    data.x = torch.tensor(data_file["x"]).type(dtype=torch.float)
    data.y = torch.tensor(data_file["y"])
    data.edge_index = torch.tensor(data_file["edge_index"])
    for i in range(num_runs):
        # create masks
        masks = np.load(mask_path + "/masks" + str(i + start_mask) + ".npz")
        test_mask = torch.tensor(masks["test_mask"])
        val_mask = torch.tensor(masks["val_mask"])
        train_mask = torch.tensor(masks["train_mask"])
        true_perc = masks["true_perc"]

        data.train_mask = train_mask
        data.test_mask = test_mask
        data.val_mask = val_mask
        data.true_perc = true_perc

        for h in hidden_channels:
            for n in num_layers:
                for r in lr:
                    for wd in weight_decay:
                        for ne in num_epochs:
                            print(
                                f'Progress: {curr_comb/tot_comb * 100:.2f}% -- Model run with mask #{i + start_mask}, using'
                                f' {h} hidden channels, {n} layers, {r} learning rate, and {wd} weight decay')
                            print("=========")
                            curr_comb += 1
                            val_loss, epochs, eval_dict = run_model(data, hidden_channels=h,
                                                                    num_layers=n, lr=r,
                                                                    weight_decay=wd, num_epochs=ne,
                                                                    display=display,
                                                                    auto_stop=auto_stop, conv_crit=conv_crit,
                                                                    patience=patience)
                            # entry = [h, n, r, wd, epochs, val_loss] + list(eval_dict.values())
                            # writer.writerow(entry)
                            # file.flush()

                            if val_loss < best_val[i]:
                                best_val[i] = val_loss
                                best_hyperparams[i] = [h, n, r, wd, epochs]
                                best_results[i] = eval_dict
    for i in range(num_runs):
        line = [i + start_mask] + best_hyperparams[i] + [best_val[i]] + list(best_results[i].values())
        writer.writerow(line)
    return best_hyperparams, best_val, best_results


def run_experiment(data_file_name: str, num_masks: int, hidden_channels=64, num_layers=3, lr=.001, weight_decay=5e-4,
                   num_epochs=100, display=True, auto_stop=False, conv_crit=1e-3, patience=10, mask_path="data"):
    """
    Use this function to run the model
    :param data_file_name: the name of the file that has the graph data you want to run the model on
    :param num_masks: the number of different masks you want to average the result over
    :param hidden_channels: number of hidden channels
    :param num_layers: number of layers in the model
    :param lr: learning rate of the optimizer
    :param weight_decay: weight decay of the optimizer
    :param num_epochs: the number of epochs to run
    :param display: whether to graph the evaluation metrics or not
    :param auto_stop: whether to autostop or use epochs
    :param conv_crit: how close two consecutive validation losses must be to be considered converged
    :param patience: how many times the convergence criteria must be broken before stopping
    :param mask_path: path to the directory that holds the masks we want to use for the experiment
    :return: The minimum validation loss, the number of epochs (if autostopped),
             a dictionary that contains all the final evaluation metrics
    """
    data_file = np.load(data_file_name)
    avg_val_loss = 0
    avg_epochs = 0
    avg_eval_dict = {"cheb": 0, "cos": 0, "canb": 0, "clark": 0, "int": 0,
                     "kl": 0}
    # Construct the data graph
    data = geo.data.Data()
    data.x = torch.tensor(data_file["x"]).type(dtype=torch.float)
    data.y = torch.tensor(data_file["y"])
    data.edge_index = torch.tensor(data_file["edge_index"])
    for n in range(num_masks):
        # create masks
        masks = np.load(mask_path + "/masks" + str(n) + ".npz")
        test_mask = torch.tensor(masks["test_mask"])
        val_mask = torch.tensor(masks["val_mask"])
        train_mask = torch.tensor(masks["train_mask"])
        true_perc = masks["true_perc"]

        data.train_mask = train_mask
        data.test_mask = test_mask
        data.val_mask = val_mask
        data.true_perc = true_perc

        val_loss, epochs, eval_dict = run_model(data, hidden_channels=hidden_channels, num_layers=num_layers, lr=lr,
                                                weight_decay=weight_decay, num_epochs=num_epochs, display=display,
                                                auto_stop=auto_stop, conv_crit=conv_crit, patience=patience)
        avg_val_loss += val_loss
        avg_epochs += epochs
        for k, v in eval_dict.items():
            avg_eval_dict[k] += v
    avg_val_loss /= num_masks
    avg_epochs /= num_masks
    for k, v in avg_eval_dict.items():
        avg_eval_dict[k] = v / num_masks
    return avg_val_loss, avg_epochs, avg_eval_dict

"""
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='dblp')
parser.add_argument('--mask_split', type=str, default='80_10_10')
parser.add_argument('--start_mask', type=int, default=0)
parser.add_argument('--num_runs', type=int, default=5)
parser.add_argument('--auto_stop', type=bool, default=True)
parser.add_argument('--add', type=bool, default=False)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--display', type=bool, default=False)

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
elif args.dataset == "imdb":
    data_path += "imdb/imdb_cc"
data_path += ".npz"

file_name = "data/" + args.dataset + "/results/hyperparameter_search_" + args.mask_split
mask_path = "data/" + args.dataset + "/masks_" + args.mask_split

opt = optimize_hyperparameters(data_path, [32, 64, 128], [2, 3, 4], [.001, .0005, .0001], [5e-3, 5e-4], add=args.add,
                         auto_stop=args.auto_stop, num_runs=args.num_runs, file_name=file_name,
                         mask_path=mask_path, patience=args.patience, display=args.display, start_mask=args.start_mask)
print(opt)
"""
# optimize_hyperparameters("data/imdb/imdb_cc.npz", [32, 64, 128], [3, 4], [.001, .0005, .0001], [5e-3, 5e-4],
#                          add=False, auto_stop=True, num_runs=5, file_name="data/imdb/results/hyperparameter_search_60_20_20",
#                          mask_path="data/imdb/masks_60_20_20", patience=10, display=False)
#
# optimize_hyperparameters("data/imdb/imdb_cc.npz", [32, 64, 128], [3, 4], [.001, .0005, .0001], [5e-3, 5e-4],
#                          add=False, auto_stop=True, num_runs=5,
#                          file_name="data/imdb/results/hyperparameter_search_40_30_30",
#                          mask_path="data/imdb/masks_40_30_30", patience=10, display=False)
#
# optimize_hyperparameters("data/dblp/APA_CC.npz", [32, 64, 128], [3, 4], [.001, .0005, .0001], [5e-3, 5e-4],
#                          add=False, auto_stop=True, num_runs=5, file_name="data/dblp/results/hyperparameter_search_80_10_10",
#                          mask_path="data/dblp/masks_80_10_10", patience=10, display=False)
#
# optimize_hyperparameters("data/dblp/APA_CC.npz", [32, 64, 128], [3, 4], [.001, .0005, .0001], [5e-3, 5e-4],
#                          add=False, auto_stop=True, num_runs=5, file_name="data/dblp/results/hyperparameter_search_50_20_30",
#                          mask_path="data/dblp/masks_50_20_30", patience=10, display=False)

# model = run_model(homo_dblp, display=False)
# print(model)
# experiment = run_experiment("data/imdb/imdb_cc.npz", 1, hidden_channels=32, num_layers=3, num_epochs=200, lr=.001,
#                             weight_decay=.005, display=True, mask_path="data/imdb/masks_80_10_10", auto_stop=True, patience=10)
# print(experiment)

# print(run_model(homo_dblp, lr=.0005, auto_stop=True))
