# Configuration
import torch
wandb = 'store_true'
batch_size = 32
lr = 0.001
epochs = 10000
hidden_channels = 16
num_layers = 2
dataset = 'ncanda'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')