import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

from components.Encoder import GCNEncoder, GATEncoder
from components.Decoder import Decoder

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        

class CNNAutoencoder(nn.Module):
    
    """ AutoEncoder module for CNN Autoencoder
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 hidden1=12,
                 hidden2=8,
                 hidden3=128
                ):
        
        super(CNNAutoencoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.encoder = CNNEncoder(latent_dims,
                                  num_nodes,
                                  bbx_size,
                                  hidden1,
                                  hidden2,
                                  hidden3,
                                 )
        self.decoder = CNNDecoder(latent_dims,
                                  num_nodes,
                                  bbx_size,
                                  hidden1,
                                  hidden2,
                                  hidden3,)
        
        
        
        
    def forward(self, X, nodes):

        latent = self.encoder(X)
        
        x_bbx = self.decoder(latent)

        return x_bbx
                        
class CNNEncoder(nn.Module):
    
    """ Encoder module for CNN Autoencoder
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 hidden1=12,
                 hidden2=8,
                 hidden3=128
                ):
        
        super(CNNEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.conv1X1 = nn.Conv2d(num_nodes, num_nodes, 1)
        self.conv1 = nn.Conv2d(num_nodes, hidden1, 3)
        self.conv2 = nn.Conv2d(hidden1, hidden2, 3)
        
    def forward(self, X , nodes):

        X = self.conv1X1(X)
        X = self.conv1(X)
        X = self.conv2(X)
        
        return X

class CNNDecoder(nn.Module):
    
    """ Decoder module for CNN Autoencoder
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 hidden1=32,
                 hidden2=16,
                 hidden3=128
                ):
        
        super(CNNDecoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.conv1 = torch.nn.Conv2d(hidden2, hidden1, 3)
        self.conv2 = torch.nn.Conv2d(hidden1, num_nodes, 3)
        self.dense_bbx = nn.Linear(num_nodes,num_nodes*bbx_size)
        self.act1 = nn.Sigmoid()

        
        
    def forward(self, latent_X , nodes,):
        
        batch_size = X.shape[0]
        X = self.conv1(latent_X)
        X = self.conv2(X)
        
        x_bbx = self.dense_box(X)
        X_bbx = self.act1(x_bbx)
        x_bbx = torch.reshape(x_bbx,[batch_size, self.num_nodes, self.bbx_size])

        return x_bbx

                        
                        