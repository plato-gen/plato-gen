import numpy as np
import torch
import torch.nn as nn

from src.layout_generation.archs.Encoder import GCNEncoder, GATEncoder
from src.layout_generation.archs.Decoder import Decoder

if torch.cuda.is_available():
    device = torch.device('cuda')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        

class GCNAutoEncoder(nn.Module):
    
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
                 dynamic_margin=False,
                 output_log=False,
                 area_encoding=False,
                 predict_edges=False,
                 predict_class=False,
                ):
        
        super(GCNAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.num_obj_classes = num_obj_classes
        self.encoder = GCNEncoder(latent_dims,
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
        self.decoder = Decoder(latent_dims,
                               num_nodes,
                               bbx_size,
                               num_obj_classes,
                               label_size,
                               output_log,
                               predict_edges,
                               predict_class
                              )
        
        
        
    def forward(self, E, X , nodes, obj_class, variational=False, predict_adj_class=False):

        z_mean, z_logvar = self.encoder(E, X, obj_class)
        batch_size = z_mean.shape[0]
        
        #sampling
#         epsilon = torch.normal(torch.zeros(z_logvar.shape, device=device))
#         z_latent = z_mean + epsilon*torch.exp(z_logvar)
        
        #sampling
        if variational:
            epsilon = torch.normal(torch.zeros(z_logvar.shape))
            z_latent = z_mean + epsilon*torch.exp(z_logvar)
        else:
            z_latent = z_mean
            
        # conditioning
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
        
        x_bbx, x_lbl, x_edge, class_pred = self.decoder(conditioned_z)
        
        if self.dynamic_margin:
            
            X_reshaped = torch.reshape(X, (batch_size, self.num_nodes, self.bbx_size+1))
            margin = self.margin_layer(torch.cat([X_reshaped[:, :, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            
            if predict_adj_class:
                return x_bbx, x_lbl, x_edge, class_pred, z_mean, z_logvar, margin
                
            return x_bbx, x_lbl, z_mean, z_logvar, margin

        if predict_adj_class:
            return x_bbx, x_lbl, x_edge, class_pred, z_mean, z_logvar
        
        return x_bbx, x_lbl, z_mean, z_logvar


class GATAutoEncoder(nn.Module):
    
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
                 hidden3=128
                ):
        
        super(GATAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.num_obj_classes = num_obj_classes
        self.encoder = GATEncoder(latent_dims,
                                  num_nodes,
                                  bbx_size,
                                  label_size,
                                  num_obj_classes,
                                  hidden1,
                                  hidden2,
                                  hidden3,
                                 )
        self.decoder = Decoder(latent_dims,
                               num_nodes,
                               bbx_size,
                               num_obj_classes,
                               label_size)
        
        
    def forward(self,E, X , nodes, obj_class):

        z_mean, z_logvar = self.encoder(E, X, obj_class)
        batch_size = z_mean.shape[0]
        
        #sampling
        epsilon = torch.normal(torch.zeros(z_logvar.shape, device=device))
        z_latent = z_mean + epsilon*torch.exp(z_logvar)
        
        # conditioning
        nodes = torch.reshape(nodes,(batch_size,self.num_nodes))
        obj_class = torch.reshape(obj_class,(batch_size,self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([obj_class, conditioned_z],dim=-1)
        
        x_bbx, x_lbl, x_edge, class_pred = self.decoder(conditioned_z)
        
        return x_bbx, x_lbl, x_edge, class_pred, z_mean, z_logvar
