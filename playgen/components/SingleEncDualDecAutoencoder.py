import numpy as np
import torch
import torch.nn as nn

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

from components.AutoEncoder import GCNAutoEncoder
from components.DenseAutoencoder import Encoder, Decoder
from components.Encoder import GCNEncoder, GATEncoder
from components.Decoder import Decoder as GCNDecoder
    
    
class SingleEncDualDecAutoencoder(nn.Module):
    
    """ AutoEncoder module for Box-Vae
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 num_obj_classes,
                 label_size=1,
                 hidden1=32,
                 hidden2=16,
                 hidden3=128,
                 dense_hidden1=8,
                 dense_hidden2=4,
                 dynamic_margin=False,
                 output_log=False,
                 area_encoding=False,
                 coupling=False,
                ):
        
        super(SingleEncDualDecAutoencoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.num_obj_classes = num_obj_classes
        self.gcn_encoder = GCNEncoder(latent_dims,
                                  num_nodes,
                                  bbx_size,
                                  label_size,
                                  num_obj_classes,
                                  hidden1,
                                  hidden2,
                                  hidden3,
                                 )
        
        self.dynamic_margin = dynamic_margin
        if self.dynamic_margin:
            self.margin_layer = nn.Linear(2*bbx_size, 2)
            self.margin_activation = nn.Sigmoid()
            
        if area_encoding:
            bbx_size-=1
        self.gcn_decoder = GCNDecoder(latent_dims,
                               num_nodes-1,
                               bbx_size,
                               num_obj_classes,
                               label_size,
                               output_log=False,
                               predict_edges=False,
                               predict_class=False,
                               object_bbox=False
                              )
        

        self.dense_decoder = Decoder(
                               latent_dims,
                               bbx_size,
                               num_obj_classes,
                               num_nodes-1,
                               dense_hidden2,
                               dense_hidden1,
                               coupling
                              )
        
        
        
    def forward(self, E, X_part, nodes, obj_class, variational=False, coupling=False):
        
        z_mean_part, z_logvar_part = self.gcn_encoder(E, X_part, obj_class)
        
        batch_size = z_mean_part.shape[0]
        
        
        #sampling
        if variational:
            epsilon_part = torch.normal(torch.zeros(z_logvar_part.shape))
            z_latent_part = z_mean_part + epsilon_part*torch.exp(z_logvar_part)
            
        else:
            z_latent_part = z_mean_part            

        # obj and part conditioning
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        nodes = nodes[:, 1:]
        conditioned_z = torch.cat([nodes, obj_class, z_latent_part],dim=-1)
        x_bbx, x_lbl, _, _ = self.gcn_decoder(conditioned_z)
        x_obj_bbx = self.dense_decoder(conditioned_z)
        
        if self.dynamic_margin:
            
            X_reshaped = torch.reshape(X_part, (batch_size, self.num_nodes, self.bbx_size+1))
            margin = self.margin_layer(torch.cat([X_reshaped[:, 1:, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            margin = torch.cat([torch.zeros(batch_size, 1, 2), margin], axis=1)
            return x_bbx, x_obj_bbx, x_lbl, z_mean_part, z_logvar_part, margin

        return x_bbx, x_obj_bbx, x_lbl, z_mean_part, z_logvar_part


