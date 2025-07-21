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
        

class DenseAutoencoder(nn.Module):
    
    """ AutoEncoder module for Dense Autoencoder
    """
    def __init__(self,
                 latent_dims,
                 bbx_size,
                 num_obj_classes,
                 num_parts,
                 hidden1=12,
                 hidden2=8,
                ):
        
        super(DenseAutoencoder, self).__init__()
        self.latent_dims = latent_dims
        self.bbx_size = bbx_size
        self.encoder = Encoder(latent_dims,
                               bbx_size,
                               hidden1,
                               hidden2,
                              )
        self.decoder = Decoder(latent_dims,
                               bbx_size,
                               num_obj_classes,
                               num_parts,
                               hidden1,
                               hidden2,
                              )
        
    def forward(self, X):

        latent = self.encoder(X)
        
        x_bbx = self.decoder(latent)

        return x_bbx
                        
class Encoder(nn.Module):
    
    """ Encoder module for Dense Autoencoder
    """
    def __init__(self,
                 latent_dims,
                 bbx_size,
                 hidden1=32,
                 hidden2=16,
                ):
        
        super(Encoder, self).__init__()
        self.latent_dims = latent_dims
        self.d1 = nn.Linear(bbx_size, hidden2)
        self.d2 = nn.Linear(hidden2, hidden1)
        self.latent = nn.Linear(hidden1, latent_dims)
        
    def forward(self, X):

        X = self.d1(X)
        X = self.d2(X)
        X = self.latent(X)
        
        return X

class Decoder(nn.Module):
    
    """ Decoder module for CNN Autoencoder
    """
    def __init__(self,
                 latent_dims,
                 bbx_size,
                 num_obj_classes,
                 num_parts,
                 hidden1=32,
                 hidden2=16,
                 coupling=False
                ):
        
        super(Decoder, self).__init__()
        if coupling:
            self.d1 = nn.Linear(2*latent_dims+num_obj_classes+num_parts, hidden1)
        else:
            self.d1 = nn.Linear(latent_dims+num_obj_classes+num_parts, hidden1)
        self.d2 = nn.Linear(hidden1, hidden2)
        self.dense_bbx = nn.Linear(hidden2,bbx_size)
        self.act1 = nn.Sigmoid()

        
        
    def forward(self, latent_X ):
        
        X = self.d1(latent_X)
        X = self.d2(X)
        x_bbx = self.dense_bbx(X)
        X_bbx = self.act1(x_bbx)
        
        return x_bbx

                        
                        