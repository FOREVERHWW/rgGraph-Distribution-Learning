# deep learning packages
#from torch_geometric.nn import GCNConv
from layers import GCNConv
import torch
import torch_geometric as geo
from torch.nn import Linear
import torch.nn.functional as F
import sklearn
from sklearn.metrics import mean_squared_error
from copy import deepcopy
# Main computation libraries
import scipy.sparse as sp
from scipy.spatial import distance
import numpy as np
import scipy as sp
import math
import random
import networkx as nx
from torch.nn import KLDivLoss,CrossEntropyLoss
# visualization
import matplotlib.pyplot as plt
from utils import *
from gcn import GCN
from parameterized_adj import PGE

eps=1e-7
class MUGCN():
    def __init__(self,dataset,train_mask,val_mask,test_mask,args,device='cuda',**kwargs):
        """
        two gcn model: gcnv, gcnl
        parameterized adj: g
        decoder for reconstructor: decoder
        original graph data: Xv, Yv,edge_indexv
        generated data: Xl,edge_indexl
        combined data:fv,fl
        training related: args
        """
        self.Xv = dataset.x.to(device)
        self.Yv = dataset.y.to(device)
        self.edge_indexv = dataset.edge_index.to(device)
        self.train_mask = train_mask.to(device)
        self.test_mask = test_mask.to(device)
        self.val_mask = val_mask.to(device)
        self.args = args
        self.device = device
        self.inner_epochs = args.inner_epochs
        self.feat_dim = dataset.x.shape[1]
        self.n = dataset.x.shape[0]
        self.m = dataset.y.shape[1]
        self.gcnv = GCN(self.feat_dim, self.m, args.gcnhidden, args.gcnvlayers).to(device)
        self.gcnl = GCN(self.feat_dim, self.m, args.gcnhidden, args.gcnllayers).to(device)
        self.g = PGE(self.feat_dim, self.m, args.pgehidden, args.pgelayers, device, args=None).to(device)
        self.g.reset_parameters()
        # self.decoder = GCN(args.gcnhidden,feat_dim, args.decoderhidden, args.decoderlayers).to(device) # TBD
        self.Xl = genfeat(self.Xv,self.Yv,self.train_mask).to(device)
        self.mode = args.mode
        if self.mode == 'dynamic':
            adjl = self.g.inference(self.Xl)
            # print(adjl)
            # adjl = (adjl>=0.5).float()
            print(adjl)
        elif self.mode== 'static':
            adjl = torch.sigmoid(torch.matmul(self.Xl,torch.t(self.Xl)))
            #adjl = torch.corrcoef(self.Xl)
            # adjg = (adjg>=0.5).double()
            # print(adjl)
        self.edge_indexl,self.edge_indexlw = from_adj_to_edge_index(adjl) # unnormalized stored as edgeindex
        self.adjl_raw = adjl
        self.adjl = gcn_norm(adjl)
        self.adjv = gcn_norm(from_edge_index_to_adj(self.edge_indexv,None)) # normalized adjv
        adjv = from_edge_index_to_adj(self.edge_indexv)
        adjlv = genhetero(adjv,adjl,self.Yv,self.train_mask)
        self.edge_indexF,self.edge_indexFw = from_adj_to_edge_index(adjlv) # unnormalized hetero edge index
        self.adjlv = gcn_norm(adjlv) # normalized hetero adj
        self.fv = torch.vstack((self.Xv,self.Xl)) # feature of node stacked
        self.fl = torch.vstack((self.Xv,self.Xl)) # feature of label stacked
        self.freq_v = args.freqv
        self.freq_l = args.freql
        if self.mode =='static':
            self.optimizer = torch.optim.Adam(list(self.gcnv.parameters())+list(self.gcnl.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = torch.optim.Adam(list(self.gcnv.parameters())+list(self.gcnl.parameters()),lr=args.lr,weight_decay=args.weight_decay)
            self.optimizer_graph = torch.optim.Adam(list(self.g.parameters()),lr=args.lr,weight_decay=args.weight_decay)
        self.epoch = args.epochs
        self.adjfv = None
    def train(self):
        with torch.autograd.set_detect_anomaly(True):
            self.gcnl.train()
            self.gcnv.train()
            if self.mode=='dynamic':
                self.g.train()
            i = 1
            epochs = self.epoch
            max_patience = self.args.patience
            # shared feature of node and labels
            fv =self.fv
            fl =self.fl
            # label distribution for node
            yv = self.Yv
            # labels for label node
            yl = torch.tensor([i for i in range(self.m)]).to(self.device) # TBD
            # node feature alone
            Xv = self.Xv
            # label feature alone
            Xl = self.Xl
            # update frequency
            freq_v = self.freq_v
            freq_l = self.freq_l
            # label edge index and weight
            edge_indexl = self.edge_indexl
            edge_indexlw = self.edge_indexlw
            # node edge index
            edge_indexv = self.edge_indexv
            edge_indexvw = None
            num_examples = Xv.shape[0]
            edge_indexF = self.edge_indexF
            edge_indexFw = self.edge_indexFw
            # normalized adj for node and label
            adjv = self.adjv
            adjl = self.adjl
            adjlv = self.adjlv
            adjl_raw = self.adjl_raw
            # initialize truncate
            adjfl = truncate(adjlv,'l',num_examples)
            adjfv = truncate(adjlv,'v',num_examples)
            # print(adjfl.shape)
            # print(adjfv.shape)
            # edge_indexFl,edge_indexFlw = from_adj_to_edge_index(adjfl)
            # edge_indexFv,edge_indexFvw = from_adj_to_edge_index(adjfv)
            # print(from_edge_index_to_adj(edge_indexFl,edge_indexFlw).shape)
            best_val_loss = 100
            patience = 0
            early_stop = False
            best_epoch = i
            val_loss_record = []
            reg_loss_record = []
            con_loss_record = []
            #Best_reg_loss = 100
            #patience_reg = 0
            while i<=epochs and not early_stop:
                #patience_reg = 0
                self.optimizer.zero_grad()
                # adjfl = truncate(from_edge_index_to_adj(self.edge_indexF,self.edge_indexFw),'l',num_examples)
                # adjfv = truncate(from_edge_index_to_adj(self.edge_indexF,self.edge_indexFw),'v',num_examples)
                # edge_indexFl,edge_indexFlw = from_adj_to_edge_index(adjfl)
                # edge_indexFv,edge_indexFvw = from_adj_to_edge_index(adjfv)
                # print(f'adj fl shape:{adjfl.shape}')
                # print(f'adj fv shape:{adjfv.shape}')
                # print(f'fv shape:{fv.shape}')
                # print(f'fl shape:{fl.shape}')
                targetl = self.gcnl.forward(fl,adjfl,adjl,'l')
                targetv = self.gcnv.forward(fv,adjfv,adjv,'v')
                # print(f'input of feature for label:{fl.shape}')
                # print(f'output of label gcn:{targetl.shape}')
                # print(f'output of node gcn:{targetv.shape}')
                # Xl = self.gcnl.get_embedding(fl,adjfl,adjl)
                # Xl_syn = genfeat(self.gcnv.get_embedding(fv,adjfv,adjv),yv,self.train_mask)
                # adjl_syn = self.g.forward(Xl_syn)
                # adjl_new = self.g.forward(Xl)
                # # adjl_syn = torch.sigmoid(torch.matmul(Xl_syn,torch.t(Xl_syn)))
                # # adjl_new = torch.sigmoid(torch.matmul(Xl,torch.t(Xl)))
                # loss_reg = self.get_pge_loss(adj_norm(adjl_new),adj_norm(adjl_syn))
                # loss_reg.backward()  # Derive gradients.
                # self.optimizer_graph.step()  # Update parameters based on gradients.
                if i % freq_v == 0:
                    if self.mode =='dynamic':
                        #self.inner_epochs = 0 
                        Best_reg_loss = 100
                        patience_reg = 0
                        for j in range(self.inner_epochs):
                            #print("inside loop")
                            self.g.train()
                            self.optimizer_graph.zero_grad()
                            Xl = self.gcnl.get_embedding(fl,adjfl,adjl)
                            Xl_syn = genfeat(self.gcnv.get_embedding(fv,adjfv,adjv),yv,self.train_mask)
                            adjl_syn = self.g.forward(Xl_syn)
                            adjl_new = self.g.forward(Xl)
                            #print(Xl_.requires_grad)
                            #print(Xl_syn.requires_grad)
                            #print(adjl_syn.requires_grad)
                            #print(adjl_new.requires_grad)
                            # adjl_syn = torch.sigmoid(torch.matmul(Xl_syn,torch.t(Xl_syn)))
                            # adjl_new = torch.sigmoid(torch.matmul(Xl,torch.t(Xl)))
                            loss_reg = self.get_pge_loss(adj_norm(adjl_new),adj_norm(adjl_syn),adj_norm(adjl_raw))
                            loss_reg.backward()  # Derive gradients.
                            reg_loss_record.append(loss_reg.item())
                            self.optimizer_graph.step()  # Update parameters based on gradients.
                            if loss_reg.item()<Best_reg_loss:
                                Best_reg_loss = loss_reg.item()
                                patience_reg = 0
                                best_g = deepcopy(self.g.state_dict())
                            else:
                                patience_reg+=1
                            if patience_reg>=70:
                                print(j)
                                print("early stop for PGE training")
                                break
                        self.g.load_state_dict(best_g)
                            #reg_loss_record.append(loss_reg.item())
                    Xl_ = self.gcnl.get_embedding(fl,adjfl,adjl)
                    Xl_syn = genfeat(self.gcnv.get_embedding(fv,adjfv,adjv),yv,self.train_mask)
                    adjl_syn = self.g.inference(Xl_)
                    adjl_new = self.g.inference(Xl_syn)
                    #loss_reg,loss_con = self.get_pge_loss_test(adj_norm(adjl_new),adj_norm(adjl_syn),adj_norm(adjl_raw))
                    #reg_loss_record.append(loss_reg.item())
                    #con_loss_record.append(loss_con.item())
                    # print(f'xl shape:{Xl_.shape}')
                    # print(f'xv shape:{Xv.shape}')
                    fv = torch.vstack((Xv,Xl_))
                    # print('fv shpae:',fv.shape)
                    self.fv = fv
                    if self.mode == 'dynamic':
                        adjl_raw = self.g.inference(Xl_)
                        # adjl_raw = (adjl_raw>0.5).float()
                        #print(adjl_raw)
                        adjl = gcn_norm(adjl_raw)
                    # adjl = torch.sigmoid(torch.matmul(Xl,torch.t(Xl)))
                        adjf = from_edge_index_to_adj(edge_indexF,edge_indexFw)
                        adjlv = gcn_norm(updatehetero(adjf,adjl_raw))
                        adjfl = truncate(adjlv,'l',num_examples)
                        adjfv = truncate(adjlv,'v',num_examples)
                    # if self.mode =='dynamic':
                    #     adjgl = self.g.inference(self.gcnl.get_embedding(fl,edge_indexFl,edge_indexl,edge_indexFlw,edge_indexlw)[self.n:])
                    elif self.mode =='static':
                        '''adjl_raw = torch.sigmoid(torch.matmul(Xl_,torch.t(Xl_)))
                        #adjl_raw = self.g.inference(Xl_)
                        adjl = gcn_norm(adjl_raw)
                        adjf = from_edge_index_to_adj(edge_indexF,edge_indexFw)
                        adjlv = gcn_norm(updatehetero(adjf,adjl_raw))
                        adjfl = truncate(adjlv,'l',num_examples)
                        adjfv = truncate(adjlv,'v',num_examples)'''
                    #     # adjgl = (adjgl>=0.5).double()
                    # edge_indexl,edge_indexlw = from_adj_to_edge_index(adjgl)
                    # adjf = from_edge_index_to_adj(self.edge_indexF,self.edge_indexFw)
                    # self.edge_indexl = edge_indexl
                    # elv,elw = from_adj_to_edge_index(updatehetero(adjf,adjgl))
                    # self.edge_indexF,self.edge_indexFw = normalize(elv,elw)
                    # adjfl = truncate(from_edge_index_to_adj(self.edge_indexF,self.edge_indexFw),'l',num_examples)
                    # adjfv = truncate(from_edge_index_to_adj(self.edge_indexF,self.edge_indexFw),'v',num_examples)
                    # edge_indexFl,edge_indexFlw = from_adj_to_edge_index(adjfl)
                    # edge_indexFv,edge_indexFvw = from_adj_to_edge_index(adjfv)
                    # Xl_syn = genfeat(self.gcnv.get_embedding(fv,edge_indexFv,edge_indexv,edge_indexFvw,None)[:self.n],yv,self.train_mask)
                    # loss2 = self.get_pge_loss(adjgl,Xl_syn)
                    # loss2.backward(retain_graph=True)
                    # self.optimizer_graph.step()
                if i % freq_l == 0:
                    fl = torch.vstack((self.gcnv.get_embedding(fv,adjfv,adjv),Xl))
                    self.fl = fl
                Xl_ = self.gcnl.get_embedding(fl,adjfl,adjl)
                Xl_syn = genfeat(self.gcnv.get_embedding(fv,adjfv,adjv),yv,self.train_mask)
                loss = self.get_loss(targetl,yl,targetv,yv,Xl_,Xl_syn)
                # if i % freq_v ==0:
                #     loss+=loss2
                loss.backward()  # Derive gradients.
                self.optimizer.step()  # Update parameters based on gradients.
                val_loss = self.get_val_loss(targetl,yl,targetv,yv).item()
                #val_loss_record.append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_fv = fv
                    best_adjfv = adjfv
                    best_adjv = adjv
                    best_model = deepcopy(self.gcnv.state_dict())
                    best_epoch = i
                    patience = 0
                else:
                    patience+=1
                if patience>=max_patience:
                    print(f"early stop triggered, best epoch:{best_epoch}")
                    early_stop = True
                print(f'progress: {i/epochs*100:.2f}%, training loss:{loss.item():.3f}')
                i+=1
            print(f"early stop triggered, best epoch:{best_epoch},best_val:{best_val_loss}")
            self.adjfv = adjfv
            if self.args.save == 1:
                #np.save('val_record.npy',np.array(val_loss_record))
                np.save('dblp_reg_record.npy',np.array(reg_loss_record))
                np.save('dblp_con_record.npy',np.array(con_loss_record))
            print(f"end of training{epochs}")
            return (best_val_loss,best_epoch,best_fv,best_adjfv,best_adjv,best_model)
    def get_loss(self,targetl,yl,targetv,yv,Xl_,Xl_syn):
        # bottom node graph loss
        training_mask = self.train_mask
        loss_v = KLDivLoss(reduction="batchmean")
        kl_loss = loss_v((targetv[training_mask]+eps).log(),(yv[training_mask]+eps))
        # print(f'predicted value:{targetv[training_mask]}')
        #print(f'KL training loss:{kl_loss:.3f}')
        # upper single label loss
        loss_l = CrossEntropyLoss()
        ce_loss = loss_l(targetl,yl)
        #print(f'CE training loss:{ce_loss:.3f}')
        # reconstruction loss, TBD
        #adjgl = torch.sigmoid(torch.matmul(Xl_,torch.t(Xl_)))
        #adj_syn = torch.sigmoid(torch.matmul(Xl_syn,torch.t(Xl_syn)))
        # reg_loss = torch.norm(get_eigen(adjgl)-get_eigen(adj_syn),p=2)
        # reg_loss = torch.linalg.matrix_norm(torch.abs(adjgl-adj_syn))
        # print("reg_loss",reg_loss)
        # regularization loss, sparse A, TBD
        loss_total = kl_loss+ce_loss
        return loss_total
    def get_pge_loss(self,adjgl,adj_syn,adjl):
        # consistency_loss = torch.linalg.matrix_norm(torch.abs(adjgl-self.g(Xl_syn)))
        consistency_loss = torch.norm(get_eigen(adjgl)-get_eigen(adj_syn),p=2)
        reg_loss = torch.norm(get_eigen(adjgl)-get_eigen(adjl),p=2)
        #print(f'consistency training loss:{(consistency_loss):.3f}')
        #print(f'reg2 training loss:{(reg_loss):.3f}')
        # total_loss = consistency_loss+self.args.alpha*reg_loss
        total_loss = consistency_loss+self.args.alpha*reg_loss
        return total_loss
    @ torch.no_grad()
    def get_pge_loss_test(self,adjgl,adj_syn,adjl):
        consistency_loss = torch.norm(get_eigen(adjgl)-get_eigen(adj_syn),p=2)
        reg_loss = torch.norm(get_eigen(adjgl)-get_eigen(adjl),p=2)
        # total_loss = consistency_loss+self.args.alpha*reg_loss
        return reg_loss,consistency_loss
    @ torch.no_grad()
    def get_val_loss(self,targetl,yl,targetv,yv):
        # bottom node graph loss
        val_mask = self.val_mask
        loss_v = KLDivLoss(reduction="batchmean")
        kl_loss = loss_v((targetv[val_mask]+eps).log(),yv[val_mask]+eps)
        # print(f'predicted value:{targetv[training_mask]}')
        # print(f'KL validation loss:{kl_loss:.4f}')
        # upper single label loss
        loss_l = CrossEntropyLoss()
        ce_loss = loss_l(targetl,yl)
        # print(f'CE validation loss:{ce_loss:.4f}')
        # reconstruction loss, TBD
        # reconstruction_loss = torch.linalg.matrix_norm(torch.abs(self.g(Xl)-adjl))
        # regularization loss, sparse A, TBD
        loss_total = kl_loss+ce_loss
        # print(f"total validation loss{loss_total:.4f}")
        return kl_loss
    def load(self,fv,adjfv,adjv,gcnv):
        self.fv = fv
        self.adjfv = adjfv
        self.adjv = adjv
        self.gcnv.load_state_dict(gcnv)
    @torch.no_grad()
    def test(self):
        self.gcnv.eval()
        # self.gcnl.eval()
        test_mask = self.test_mask
        fv = self.fv
        yv = self.Yv
        adjfv = self.adjfv
        adjv = self.adjv
        targetv = self.gcnv.forward(fv,adjfv,adjv,'v')
        #print("predicted")
        #print(targetv[test_mask].shape)
        predicted_argmax = torch.argmax(targetv[test_mask], dim=1)
        #print(predicted_argmax.shape)
        #print(predicted_argmax)
        #print("original")
        orig_arg = torch.argmax(yv[test_mask], dim=1)
        #print(orig_arg)
        acc_sum = sum(orig_arg == predicted_argmax) / predicted_argmax.shape[0]
        print("acc",acc_sum.item())
        f1_score = sklearn.metrics.f1_score(orig_arg.cpu().detach().numpy(), predicted_argmax.cpu().detach().numpy(), average="weighted")
        print(
            f'F1 Score: {f1_score}')
        loss_v = KLDivLoss(reduction="batchmean")
        # print(targetv[test_mask].log())
        # print((yv[test_mask]+eps).log())
        kl_loss = loss_v((targetv[test_mask]+eps).log(),yv[test_mask]+eps)
        kl_loss2 = loss_v((yv[test_mask]+eps).log(),targetv[test_mask]+eps)
        print(f'kl test loss:{kl_loss.item():.4f}')
        #print(f'kl test loss:{kl_loss2.item():.4f}')
        yv = yv[test_mask]+eps
        targetv = targetv[test_mask]+eps
        score_cosin = 0
        score_can = 0
        score_cheb = 0
        for i in range(yv.shape[0]):
            score_cosin += distance.cosine(yv[i].cpu().detach().numpy(),targetv[i].cpu().detach().numpy())
            score_can += distance.canberra(yv[i].cpu().detach().numpy(),targetv[i].cpu().detach().numpy())
            score_cheb += distance.chebyshev(yv[i].cpu().detach().numpy(),targetv[i].cpu().detach().numpy())
        score_cosin/= yv.shape[0]
        score_can/=yv.shape[0]
        score_cheb/=yv.shape[0]
        score_clark = clark(targetv+eps,yv+eps)
        score_intersection = intersection(targetv,yv)
        print(f'cosin distance:{score_cosin:.4f}')
        print(f'canberra distance:{score_can:.4f}')
        print(f'chebyshev distance:{score_cheb:.4f}')
        print(f'clark distance:{score_clark:.4f}')
        print(f'intersection distance:{score_intersection:.4f}')
        return np.array([kl_loss.item(),kl_loss2.item(),score_cosin,score_can,score_cheb,score_clark,score_intersection,acc_sum,f1_score])






