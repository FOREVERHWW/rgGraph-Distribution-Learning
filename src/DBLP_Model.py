# deep learning packages
from torch_geometric.nn import GCNConv
import torch
import torch_geometric as geo
from torch.nn import Linear
import torch.nn.functional as F
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Main computation libraries
import scipy.sparse as sp
from scipy.spatial import distance
import numpy as np
import scipy as sp
from scipy.stats import entropy
import math
import random

# LOSS FUNCTIONS #
EPS = 1e-7

def sum_square_loss(prediction, actual):
    """
    calculates the sum square loss of a distribution label dataset
    :param prediction: a tensor of predicted distribution labels (num_nodes x num_labels size tensor)
    :param actual: a tensor of the actual distribution labels (num_nodes x num_labels size tensor)
    :return: the average sum squared distance between all the nodes
    """
    num_nodes = prediction.size(dim=0)
    avg_ssl = 0
    for node in range(num_nodes):
        ssl = 0
        for label in range(prediction.size(dim=1)):
            ssl += (prediction[node][label].item() - actual[node][label].item()) ** 2
        avg_ssl += ssl
        # print(f' The SSL of the {node}th node: {avg_ssl}')
    avg_ssl /= num_nodes
    return torch.tensor(avg_ssl)


def kl_divergence(prediction, actual):
    """
    calculates the KL-Divergence of a distribution label dataset
    :param prediction: a tensor of predicted distribution labels (num_nodes x num_labels size tensor)
    :param actual: a tensor of the actual distribution labels (num_nodes x num_labels size tensor)
    :return: the average KL-Divergence of all the nodes
    """
    # TODO: implement the function
    pass


def clark(predicted: np.array, actual: np.array):
    """
    computes the Clark distance between two distributions
    :param predicted: the predicted distribution
    :param actual: the actual distribution
    :return: the Clark distance between two distributions
    """
    res = 0
    for i in range(np.shape(predicted)[0]):
        res += (predicted[i] - actual[i]) ** 2 / (predicted[i] + actual[i]) ** 2
    return math.sqrt(res)


def intersection(predicted: np.array, actual: np.array):
    """
    computes the intersection distance between two distributions
    :param predicted: the predicted distribution
    :param actual: the actual distribution
    :return: the intersection distance between two distributions
    """
    res = 0
    for i in range(np.shape(predicted)[0]):
        res += min(predicted[i], actual[i])
    return res


def canberra(predicted: np.array, actual: np.array):
    """
    computes the Canberra distance between two distributions
    :param predicted: the predicted distribution
    :param actual: the actual distribution
    :return: the Canberra distance between two distributions
    """
    res = 0
    for i in range(np.shape(predicted)[0]):
        res += abs(predicted[i] - actual[i]) / (predicted[i] + actual[i])
    return res


def chebyshev(predicted: np.array, actual: np.array):
    """
    computes the Chebyshev distance between two distributions
    :param predicted: the predicted distribution
    :param actual: the actual distribution
    :return: the Chebyshev distance between two distributions
    """
    res = 0
    for i in range(np.shape(predicted)[0]):
        res = max(res, abs(predicted[i] - actual[i]))
    return res


# MODEL #


class GCN(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels, layers=2, lr=.001, weight_decay=5e-4):
        """
        initializes a GCN with two layers
        :param input_dim: the dimension of your input
        :param input_dim: the dimension of your output
        :param hidden_channels: the amount of channels in the layers that aren't input or output
        :param layers: the number of layers you want in your model
        """
        super().__init__()
        # seed used so this model trains the same way everytime
        torch.manual_seed(1234567)
        self.layers = []
        self.lr = lr
        self.weight_decay = weight_decay
        for i in range(layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_channels))
            elif i == layers - 1:
                self.layers.append(GCNConv(hidden_channels, output_dim))
            else:
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers = torch.nn.ModuleList(self.layers)

    def forward(self, x, edge_index):
        """
        Used to progress the model
        :param x: feature matrix
        :param edge_index: edge matrix
        :return:
        """
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i != len(self.layers) - 1:
                x = x.relu()
                x = F.dropout(x, p=0, training=self.training)

        return F.softmax(x, dim=1)


def train(model, input_data, loss_function):
    """
    used to train the model using the inputted loss function
    :param model: the model being trained
    :param input_data: the original data used to generate the latent features
    :param loss_function: the loss function you want to use during training
    :return: training loss
    """

    model.train()
    mask = input_data.train_mask

    optimizer = torch.optim.Adam(model.parameters(), lr=model.lr, weight_decay=model.weight_decay)
    optimizer.zero_grad()  # Clear gradients.
    out = model(input_data.x, input_data.edge_index)  # Perform a single forward pass.
    loss = loss_function((out[mask] + EPS).log(),
                         input_data.y[mask] + EPS)  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


@torch.no_grad()
def calc_val_loss(model, input_data, loss_function):
    """
    Used to calculate the model's validation loss
    :param model: the model being trained
    :param input_data: the original data used to generate the latent features
    :param loss_function: the loss function you want to use to calculate loss
    :return: validation loss
    """
    model.eval()

    mask = input_data.val_mask
    out = model(input_data.x, input_data.edge_index)  # Perform a single forward pass.
    loss = loss_function((out[mask] + EPS).log(),
                         input_data.y[mask] + EPS)  # Compute the loss solely based on the training nodes.
    return loss


@torch.no_grad()
def test(model, input_data, mask, eval_func=distance.chebyshev, latent_features=None, lin=False):
    """
    used to test the given model on the masked data using eval_func
    :param model: the model to be tested
    :param mask: the mask to be used on the data
    :param eval_func: the function used to evaluate the data
    :param latent_features: latent features to test against
    :param lin: whether the model to be tested is linear or not
    :return: the average Chebyshev Distance of the model
    """
    model.eval()

    if latent_features == None:
        if lin:
            out = model(input_data.x)
        else:
            out = model(input_data.x, input_data.edge_index)
    else:
        if lin:
            out = model(latent_features)
        else:
            out = model(latent_features, input_data.edge_index)
    length = int(mask.sum())
    distance = 0
    #
    # print("predicted")
    # print(out[mask])
    predicted_argmax = torch.argmax(out[mask], dim=1)
    # print(predicted_argmax.shape)
    # print(predicted_argmax)
    # print("original")
    # print(input_data.y[mask])
    #
    orig_arg = torch.argmax(input_data.y[mask], dim=1)
    # print(orig_arg)
    print(sum(orig_arg == predicted_argmax) / predicted_argmax.shape[0])
    print(f'F1 Score: {sklearn.metrics.f1_score(orig_arg.cpu().detach().numpy(), predicted_argmax.cpu().detach().numpy(), average="weighted")}')

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    # print(f"KL Loss: {kl_loss((out[mask]).log(), input_data.y[mask])}")
    for i in range(mask.size(dim=0)):
        if mask[i]:
            distance += eval_func(input_data.y[i].cpu().detach().numpy(), out[i].cpu().detach().numpy())
    # print(out[:10])
    # print("test eps:",distance/length)
    return distance/length






