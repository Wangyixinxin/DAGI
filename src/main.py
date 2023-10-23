import argparse
import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import MLP, GINConv, global_add_pool, GCNConv
from torch.nn.functional import relu
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
import torch_geometric.transforms as T
from model import *
from util import *
from config import *

# Data Preparation
def prepare_data(dataset, batch_size):
    train_dataset = dataset[len(dataset)//5:]
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    test_dataset = dataset[:len(dataset)//5]
    test_loader = DataLoader(test_dataset, batch_size)

    
    return train_loader, test_loader


# Utilities
class TorchStandardScaler:
    def fit(self, x):
        self.mean = torch.mean(x, dim=0)
        self.std = torch.std(x, dim=0)

    def transform(self, x):
        x = torch.sub(x, self.mean)
        x = torch.div(x, self.std + 1e-7)
        return x

# Training
def train(model, loader, optimizer, ava_featindex, impute_featindex, rec_lambda, device):
    model.train()

    total_loss = 0
    recon_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        out, recon, em = model(data.x[:,ava_featindex], data.edge_index, data.batch)
        loss_cls = F.cross_entropy(out, data.y)

        loss_rec = torch.sqrt(torch.mean((recon-data.x[:,impute_featindex])**2)) 
        loss = loss_cls + loss_rec * rec_lambda
        loss.backward()
        optimizer.step()
        
        total_loss += float(loss) * data.num_graphs
        recon_loss += float(loss_rec * rec_lambda) * data.num_graphs
        
    return total_loss / len(loader.dataset), recon_loss / len(loader.dataset)

# Testing
def test(model, loader, ava_featindex, impute_featindex, rec_lambda, device):
    model.eval()
    total_correct = 0
    total_loss = 0
    recon_loss = 0
    pred_all = []
    y_all = []
    
    for data in loader:
        data = data.to(device)
        out, recon, em = model(data.x[:,ava_featindex], data.edge_index, data.batch)
        
        loss_cls = F.cross_entropy(out, data.y)
        loss_rec = torch.sqrt(torch.mean((recon-data.x[:,impute_featindex])**2))
        loss = loss_cls
        pred = out.argmax(dim=-1)
        
        total_correct += int((pred == data.y).sum())
        pred_all += pred.tolist()
        y_all += data.y.tolist()
        total_loss += float(loss) * data.num_graphs
        recon_loss += float(loss_rec * rec_lambda) * data.num_graphs
        
        total_bcorrect = balanced_accuracy_score(y_all, pred_all)
        
    return total_loss / len(loader.dataset), total_bcorrect, recon_loss / len(loader.dataset)


if __name__ == "__main__":
    
    init_wandb(name=f'GIN-{dataset}', batch_size=batch_size, lr=lr, epochs=epochs, hidden_channels=hidden_channels, num_layers=num_layers, device=device)
    
    # Data 
    # Initialization
    regionaparc_name_list, regionaseg_name_list, region_name_dict = initialize_variables()
    
    # Build Adjacency Matrices
    adjacency_matrix, region_name_list = build_adjacency_matrices(regionaparc_name_list, regionaseg_name_list)
    
    # Load Features
    file_path = '/Users/xxx.csv'  ## change it to your file
    ncanda_allfeat, G_label = load_features(file_path)
    G_list = create_graph(ncanda_allfeat, G_label, region_name_list, adjacency_matrix)
    train_loader, test_loader = prepare_data(G_list, batch_size)
    
    # Model, optimizer, and stats setup
    num_classes = 2
    rec_lambda = 1
    impute = True
    ava_featindex = [0,1,2]
    impute_featindex = [3]
    num_features = len(ava_featindex)
    
    model = GINWithJK_rec(num_features, hidden_channels, num_classes, num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    accuracy_stats = {
        'train': [],
        "test": []
    }
    loss_stats = {
        'train': [],
        "test": []
    }
    recloss_stats = {
        'train': [],
        "test": []
    }

    # Training loop
    for epoch in range(1, epochs + 1):
        loss = train(model, train_loader, optimizer, ava_featindex, impute_featindex, rec_lambda, device)
        
        train_loss, train_acc, train_rec = test(model, train_loader, ava_featindex, impute_featindex, rec_lambda, device)
        test_loss, test_acc, test_rec = test(model, test_loader, ava_featindex, impute_featindex, rec_lambda, device)
        log(Epoch=epoch, Train_loss=train_loss, Test_loss=test_loss, Train_acc=train_acc, Test_acc=test_acc, Train_rec=train_rec, Test_rec=test_rec)
        loss_stats['train'].append(train_loss)
        loss_stats['test'].append(test_loss)
        recloss_stats['train'].append(train_rec)
        recloss_stats['test'].append(test_rec)
        accuracy_stats['train'].append(train_acc)
        accuracy_stats['test'].append(test_acc)
