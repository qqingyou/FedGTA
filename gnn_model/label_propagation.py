import copy
import torch
import random
import platform
import torch.nn.functional as F

from data.utils import idx_to_mask
from utils import csr_sparse_dense_matmul
from utils import homo_adj_to_symmetric_norm


class NonParaLP():
    def __init__(self, prop_steps, num_class, subgraph, device, alpha=0.5, r=0.5, temperature=20):
        self.prop_steps = prop_steps
        self.r = r
        self.num_class = num_class
        self.alpha = alpha
        self.subgraph = subgraph.to(device)
        self.y = subgraph.y.to(device)
        self.device = device
        
        # Check if dataset has unknown nodes (e.g., Elliptic dataset with -1 labels)
        has_unknown_nodes = hasattr(subgraph, 'known_mask') or torch.any(self.y == -1)
        
        if has_unknown_nodes:
            # Filter out unknown nodes to prevent overflow and index errors
            if hasattr(subgraph, 'known_mask'):
                self.known_mask = subgraph.known_mask.to(device)
            else:
                # Create known_mask for nodes that don't have -1 labels
                self.known_mask = (self.y != -1)
            
            # Only work with known nodes
            known_indices = torch.where(self.known_mask)[0]
            
            # Filter train/val/test indices to only include known nodes
            train_known = self.subgraph.train_idx & self.known_mask
            val_known = self.subgraph.val_idx & self.known_mask  
            test_known = self.subgraph.test_idx & self.known_mask
            
            num_nodes = torch.sum(self.known_mask).item()
            train_idx_list = torch.where(train_known)[0].cpu().numpy().tolist()
            
            # Create labels only for known nodes, map unknown nodes to 0 temporarily for one_hot
            y_for_onehot = self.y.clone()
            y_for_onehot[~self.known_mask] = 0
            self.label = F.one_hot(y_for_onehot.view(-1), self.num_class).to(torch.float).to(self.device)
            # Zero out labels for unknown nodes
            self.label[~self.known_mask] = 0.0
            
            self.unlabel_idx = val_known | test_known
        else:
            # Original logic for datasets without unknown nodes (Cora, CiteSeer, PubMed)
            self.known_mask = None
            num_nodes = len(self.subgraph.train_idx)
            train_idx_list = torch.where(self.subgraph.train_idx == True)[0].cpu().numpy().tolist()
            self.label = F.one_hot(self.y.view(-1), self.num_class).to(torch.float).to(self.device)
            self.unlabel_idx = self.subgraph.val_idx | self.subgraph.test_idx
        
        # Common processing for both cases
        random.shuffle(train_idx_list)
        self.label_idx = idx_to_mask(train_idx_list, len(self.y)).to(device)
        self.temperature = temperature
        self.adj = homo_adj_to_symmetric_norm(self.subgraph.adj, r=r)

    def preprocess(self, soft_label):
        if self.known_mask is not None:
            # Only update labels for known unlabeled nodes
            unlabel_known = self.unlabel_idx & self.known_mask
            if torch.sum(unlabel_known) > 0:
                unlabel_init = soft_label[unlabel_known].to(self.device)
                self.label[unlabel_known] = unlabel_init
        else:
            # Original logic for datasets without unknown nodes
            unlabel_init = soft_label[self.unlabel_idx].to(self.device)
            self.label[self.unlabel_idx] = unlabel_init

    def eval(self):
        pred = self.output_2.max(1)[1].type_as(self.subgraph.y)
        
        if self.known_mask is not None:
            # Only evaluate on known nodes for datasets with unknown nodes
            known_pred = pred[self.known_mask]
            known_y = self.subgraph.y[self.known_mask]
            tot_correct = known_pred.eq(known_y).double()
            tot_correct = tot_correct.sum()
            tot_reliability_acc = (tot_correct / known_y.shape[0]).item()
            
            # For label_idx, only consider the intersection with known nodes
            label_and_known = self.label_idx & self.known_mask
            if torch.sum(label_and_known) > 0:
                correct = pred[label_and_known].eq(self.subgraph.y[label_and_known]).double()
                correct = correct.sum()
                reliability_acc = (correct / torch.sum(label_and_known).item()).item()
            else:
                reliability_acc = 0.0
        else:
            # Original logic for datasets without unknown nodes
            tot_correct = pred.eq(self.subgraph.y).double()
            tot_correct = tot_correct.sum()
            tot_reliability_acc = (tot_correct / self.subgraph.y.shape[0]).item()
            correct = pred[self.label_idx].eq(self.subgraph.y[self.label_idx]).double()
            correct = correct.sum()
            reliability_acc = (correct / self.subgraph.y[self.label_idx].shape[0]).item()
        
        return tot_reliability_acc, reliability_acc

    def init_lp_propagate(self, feature, init_label, alpha):
        init_label_ = copy.deepcopy(init_label.cpu())
        feature = feature.cpu().numpy()
        feat_temp = feature
        for _ in range(self.prop_steps):
            if platform.system() == "Linux":
                feat_temp = csr_sparse_dense_matmul(self.adj, feat_temp)
            else:
                feat_temp = self.adj.dot(feat_temp)

            feat_temp = alpha * feat_temp + (1 - alpha) * feature
            feat_temp[init_label_] += feature[init_label_]
        return torch.tensor(feat_temp)

    def propagate(self):
        self.output = self.init_lp_propagate(self.label, init_label=self.label_idx, alpha=self.alpha).to(self.device)
        self.output_raw = F.softmax(self.output, dim=1)
        self.output_dis = F.softmax(self.output/self.temperature, dim=1)
        self.output_raw[self.label_idx] = self.label[self.label_idx]
        self.output_dis[self.label_idx] = self.label[self.label_idx]
        return self.output_raw, self.output_dis
