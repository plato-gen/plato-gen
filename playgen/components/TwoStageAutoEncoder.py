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
                 dense_hidden1=8,
                 dense_hidden2=4,
                 dynamic_margin=False,
                 output_log=False,
                 area_encoding=False,
                 coupling=False,
                 obj_bbx_conditioning=False,
                 use_fft_on_bbx=False,
                 use_gcn_in_decoder=False,
                ):
        
        super(TwoStageAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.obj_latent_dims = 2
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.obj_bbx_conditioning = obj_bbx_conditioning
        self.num_obj_classes = num_obj_classes
        self.use_fft_on_bbx = use_fft_on_bbx
        self.use_gcn_in_decoder = use_gcn_in_decoder
        self.gcn_encoder = GCNEncoder(latent_dims,
                                  num_nodes,
                                  bbx_size * (2 if use_fft_on_bbx else 1),
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
                               num_nodes,
                               bbx_size,
                               num_obj_classes,
                               label_size,
                               output_log=False,
                               predict_edges=False,
                               predict_class=False,
                               object_bbox=True,
                               use_gcn=self.use_gcn_in_decoder
                              )
        
        if not self.obj_bbx_conditioning:
            self.dense_encoder = Encoder(self.obj_latent_dims,
                                   bbx_size,
                                   dense_hidden1,
                                   dense_hidden2,
                                  )
            self.dense_decoder = Decoder(
                                   self.obj_latent_dims,
                                   bbx_size,
                                   num_obj_classes,
                                   num_nodes,
                                   dense_hidden2,
                                   dense_hidden1,
                                   coupling
                                  )
        
        
        
    def forward(self, E, X_part, X_obj , nodes, obj_class, variational=False, coupling=False,
                obj_bx_conditioning = False, training=False, refine_iter=1):
        
        if self.use_fft_on_bbx:
            X_label, X_box = torch.split(X_part, (1, 4), dim=-1)
            X_part_fft = torch.cat([torch.sin(X_box), torch.cos(X_box), X_label], dim=-1)
            z_mean_part, z_logvar_part = self.gcn_encoder(E, X_part_fft, obj_class)
        else:
            z_mean_part, z_logvar_part = self.gcn_encoder(E, X_part, obj_class)
        
        batch_size = z_mean_part.shape[0]
        
        if self.obj_bbx_conditioning:
            z_mean_obj = X_obj
            z_logvar_obj = X_obj
            
        else:
            z_mean_obj, z_logvar_obj = self.dense_encoder(X_obj) 
        
        
        #sampling
        if variational or training:
            epsilon_obj = torch.normal(torch.zeros(z_logvar_obj.shape))
            epsilon_part = torch.normal(torch.zeros(z_logvar_part.shape))
            z_latent_part = z_mean_part + epsilon_part*torch.exp(z_logvar_part)
            z_latent_obj = z_mean_obj + epsilon_obj*torch.exp(z_logvar_obj)
        else:
            z_latent_part = z_mean_part        
        
        # obj conditioning
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        conditioned_obj_latent = torch.cat([obj_class, z_mean_obj],dim=-1)
        
        # part conditioning
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        conditioned_obj_latent = torch.cat([nodes, conditioned_obj_latent],dim=-1)
        
        # object and part representation concat
        conditioned_z = torch.cat([conditioned_obj_latent, z_latent_part],dim=-1)

        if self.use_gcn_in_decoder:
            x_bbx, x_lbl, _, _, x_bbx_refined = self.gcn_decoder(conditioned_z, E, training, refine_iter, labels=X_label)
        else:
            x_bbx, x_lbl, _, _ = self.gcn_decoder(conditioned_z)
            x_bbx_refined = None
        
        if self.obj_bbx_conditioning:
            x_obj_bbx = X_obj
        
        elif coupling:
            
            x_obj_bbx = self.dense_decoder(conditioned_z)
           
        else:
            x_obj_bbx = self.dense_decoder(conditioned_obj_latent)
        
        if self.dynamic_margin:
            
            X_reshaped = torch.reshape(X_part, (batch_size, self.num_nodes, self.bbx_size+1))
            margin = self.margin_layer(torch.cat([X_reshaped[:, :, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            return x_bbx, x_obj_bbx, x_lbl, z_mean_part, z_logvar_part, margin, z_mean_obj, z_logvar_obj, x_bbx_refined

        return x_bbx, x_obj_bbx, x_lbl, z_mean_part, z_logvar_part, z_mean_obj, z_logvar_obj, x_bbx_refined


