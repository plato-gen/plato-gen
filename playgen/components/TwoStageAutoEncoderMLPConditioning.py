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
    
    
class TwoStageAutoEncoder(nn.Module):
    
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
                 hidden_obj_conditioning=8,
                 hidden_part_conditioning=8,
                 dense_hidden1=16,
                 dense_hidden2=32,
                 dynamic_margin=False,
                 output_log=False,
                 area_encoding=False,
                 transform_conditioning=True
                ):
        
        super(TwoStageAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.num_obj_classes = num_obj_classes
        self.transform_conditioning = transform_conditioning
        self.gcn_encoder = GCNEncoder(latent_dims,
                                  num_nodes,
                                  bbx_size,
                                  label_size,
                                  num_obj_classes,
                                  hidden1,
                                  hidden2,
                                  hidden3,
                                 )
        
        self.dense_encoder = Encoder(latent_dims,
                               bbx_size,
                               hidden1,
                               hidden2,
                              )
        
        if area_encoding:
            bbx_size-=1
            
        self.dynamic_margin = dynamic_margin
        if self.dynamic_margin:
            self.margin_layer = nn.Linear(2*bbx_size, 2)
            self.margin_activation = nn.Sigmoid()
            
        self.gcn_decoder = GCNDecoder(latent_dims,
                               num_nodes,
                               bbx_size,
                               num_obj_classes,
                               label_size,
                               output_log=False,
                               predict_edges=False,
                               predict_class=False,
                               object_bbox=True,
                               hidden_obj_conditioning=8,
                               hidden_part_conditioning=8
                              )
        
        if transform_conditioning:
            self.hidden_obj_cond_mlp = nn.Linear(num_obj_classes, hidden_obj_conditioning)
            self.hidden_part_cond_mlp = nn.Linear(num_nodes, hidden_part_conditioning)
            num_obj_classes = hidden_obj_conditioning
            num_nodes = hidden_part_conditioning

        self.dense_decoder = Decoder(latent_dims,
                               bbx_size,
                               num_obj_classes,
                               num_nodes,
                               hidden1,
                               hidden2,
                              )
        
        
        
    def forward(self, E, X_part, X_obj , nodes, obj_class, variational=False):
        
        z_mean, z_logvar = self.gcn_encoder(E, X_part, obj_class)
        
        batch_size = z_mean.shape[0]
        latent_obj = self.dense_encoder(X_obj)
        
        # obj conditioning
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        if self.transform_conditioning:
            obj_class = self.hidden_obj_cond_mlp(obj_class)
        conditioned_obj_latent = torch.cat([obj_class, latent_obj],dim=-1)
        
        # part conditioning
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        if self.transform_conditioning:
            nodes = self.hidden_part_cond_mlp(nodes)
        conditioned_obj_latent = torch.cat([nodes, conditioned_obj_latent],dim=-1)
        x_obj_bbx = self.dense_decoder(conditioned_obj_latent)
        
        # sampling
#         if variational:
#             epsilon = torch.normal(torch.zeros(z_logvar.shape))
#             z_latent = z_mean + epsilon*torch.exp(z_logvar)
#         else:
            
        z_latent = z_mean            
        
        # conditioning mixing
        conditioned_z = torch.cat([conditioned_obj_latent, z_latent],dim=-1)
        
        x_bbx, x_lbl, _, _ = self.gcn_decoder(conditioned_z)
        
        if self.dynamic_margin:
            
            X_reshaped = torch.reshape(X_part, (batch_size, self.num_nodes, self.bbx_size+1))
            margin = self.margin_layer(torch.cat([X_reshaped[:, :, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            return x_bbx, x_obj_bbx, x_lbl, z_mean, z_logvar, margin

        return x_bbx, x_obj_bbx, x_lbl, z_mean, z_logvar


