from torch_geometric.nn import JumpingKnowledge, SAGEConv, global_mean_pool, AttentionalAggregation
import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d as BN
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import MLP, GINConv, global_add_pool, GCNConv

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)
            
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            #print(x.dtype, edge_index.dtype)
            x = conv(x, edge_index).relu()
            
        x = global_add_pool(x, batch)
        return self.mlp(x)
    
class GIN_rec(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels
        """
        self.convs = torch.nn.ModuleList()    
        self.convs.append(
                    Sequential(
                        Linear(in_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                    ))
        """    
        self.mlp = MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

        self.convs_rec = torch.nn.ModuleList()    
        self.convs_rec.append(
                    Sequential(
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, hidden_channels),
                        ReLU(),
                        Linear(hidden_channels, 4),
                        ReLU(),
                    ))
            
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            #print(x.dtype, edge_index.dtype)
            x = conv(x, edge_index).relu()
            
        ### recon ###
        for conv in self.convs_rec:
            x_rec = conv(x) 
            
        x = global_add_pool(x, batch)
        return self.mlp(x), x_rec
    
class GlobalAttentionNet(torch.nn.Module):
    def __init__(self, num_features, hidden, out_channels, num_layers):
        super().__init__()
        self.conv1 = SAGEConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.att = AttentionalAggregation(Linear(hidden, 1))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.att.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.att(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.sigmoid(x)

    def __repr__(self):
        return self.__class__.__name__
    
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden, out_channels, num_layers):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(hidden, hidden))
        self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
    
class GINWithJK(torch.nn.Module):
    def __init__(self, num_features, hidden, out_channels, num_layers, mode='cat'):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        em = x
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1),em

    def __repr__(self):
        return self.__class__.__name__
    
from torch_geometric.nn import (
    ASAPooling,
    GraphConv,
    JumpingKnowledge,
    global_mean_pool,
)

    
### with both classification and imputuation tasks
class GINWithJK_rec(torch.nn.Module):
    def __init__(self, num_features, hidden, out_channels, num_layers, mode='cat'):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(
                Linear(num_features, hidden),
                ReLU(),
                BN(hidden),
                Linear(hidden, hidden),
                ReLU(),
                BN(hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                        Linear(hidden, hidden),
                        ReLU(),
                        BN(hidden),
                    ), train_eps=True))
            
        self.convs_rec = torch.nn.ModuleList()    
        for i in range(num_layers - 1):
            self.convs_rec.append(
                GINConv(
                    Sequential(
                        Linear(hidden*num_layers, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        ReLU(),
                        Linear(hidden, hidden),
                        BN(hidden),
                        ReLU(),
                        Linear(hidden, 1),
                    ), train_eps=True))   
        
        self.jump = JumpingKnowledge(mode)
        if mode == 'cat':
            self.lin1 = Linear(num_layers * hidden, hidden)
        else:
            self.lin1 = Linear(hidden, hidden)
        self.lin2 = Linear(hidden, out_channels)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.jump.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch=None):
        x = self.conv1(x, edge_index)
        xs = [x]
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]
        x = self.jump(xs)
        em = x
        #print(x.shape)
        
        ### recon ###
        i = 0
        for conv in self.convs_rec:
            i += 1
            #print("convs_rec",i)
            x_rec = conv(x, edge_index)
        
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        rec = x
        return F.log_softmax(x, dim=-1), x_rec, em

    def __repr__(self):
        return self.__class__.__name__
