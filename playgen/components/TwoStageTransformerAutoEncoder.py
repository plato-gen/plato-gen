import numpy as np
import math
import torch
from torch import nn, Tensor

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

from torch.nn import TransformerEncoder, TransformerEncoderLayer
from components.Decoder import Decoder
from components.DenseAutoencoder import Encoder, Decoder

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
        
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
        
    
class TransformerAutoEncoder(nn.Module):
    
    """ AutoEncoder module for Box-Vae
        TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, dropout)
        (self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5)
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 num_obj_classes,
                 label_size=1,
                 hidden1=32,
                 hidden2=16,
                 hidden3=32,
                 dropout=0.1,
                 nhead=3,
                 dynamic_margin=False,
                 output_log=False,
                 area_encoding=False
                ):
        
        super(TransformerAutoEncoder, self).__init__()
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.num_obj_classes = num_obj_classes
        
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(bbx_size, dropout, num_nodes)
        encoder_layers = TransformerEncoderLayer(
            d_model=bbx_size,
            nhead=nhead,
            dim_feedforward=latent_dims,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, hidden1)
        #self.encoder = nn.Embedding(int(num_nodes), int(bbx_size-label_size))
        
        self.dense_boxes = nn.Linear(bbx_size, hidden2)
        self.dense_labels = nn.Linear(label_size,hidden2)
        self.act = nn.ReLU()
        # self.dense_enc1 = nn.Linear(
        self.dense1 = nn.Linear(hidden2*num_nodes,hidden3)
        self.dense2 = nn.Linear(bbx_size*num_nodes+num_obj_classes,hidden3)
        self.dense3 = nn.Linear(hidden3,hidden3)
        
        
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
                               output_log
                              )
                
        self.latent = nn.Linear(hidden3,latent_dims)
        self.dense_encoder = Encoder(latent_dims,
                               bbx_size,
                               hidden1,
                               hidden2,
                              )
        self.dense_decoder = Decoder(latent_dims,
                               bbx_size,
                               num_obj_classes,
                               num_nodes,
                               hidden1,
                               hidden2
                              )
        
        
    def forward(self, E, X, X_obj, nodes, class_labels, variational=False):

        batch_size = int(X.shape[0]/self.num_nodes)
        
        latent_obj = self.dense_encoder(X_obj)
        
        #obj conditioning
        obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
        conditioned_obj_latent = torch.cat([obj_class, latent_obj],dim=-1)
        
        src = torch.reshape(X, (self.num_nodes, batch_size, X.shape[-1])) 
        src = self.pos_encoder(src[:, :, 1:])
        
        output = self.transformer_encoder(src)
        
        x = torch.reshape(output,(batch_size,self.num_nodes*src.shape[-1]))
        
        boxes = X[:, 1:]
        boxes = torch.reshape(boxes,(batch_size,self.num_nodes,boxes.shape[-1]))                  
        boxes = self.act(self.dense_boxes(boxes))
        
        node_labels = X[:,:1]
        node_labels = torch.reshape(node_labels,(batch_size,self.num_nodes,node_labels.shape[-1]))                  
        node_labels = self.act(self.dense_labels(node_labels))
        
        mix = torch.add(boxes,node_labels)
        mix = torch.reshape(mix,(batch_size,mix.shape[-2]*mix.shape[-1]))                  
        mix = self.act(self.dense1(mix))
        
        class_labels = torch.reshape(class_labels,(batch_size,int(class_labels.shape[-1]/batch_size)))
        x = torch.cat([class_labels,x],dim=-1)    
        
        x = self.act(self.dense2(x))
        x = torch.add(x,mix)
        
        x = self.act(self.dense3(x))
        
        z_mean = self.act(self.latent(x))
        z_logvar = self.act(self.latent(x))
        
        
        batch_size = z_mean.shape[0]
        
        #sampling
        if variational:
            epsilon = torch.normal(torch.zeros(z_logvar.shape, device=device))
            z_latent = z_mean + epsilon*torch.exp(z_logvar)
        else:
            z_latent = z_mean
        
        # conditioning
        nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
        class_labels = torch.reshape(class_labels, (batch_size, self.num_obj_classes))
        conditioned_z = torch.cat([nodes, z_latent],dim=-1)
        conditioned_z = torch.cat([class_labels, conditioned_z],dim=-1)
        
        x_bbx, x_lbl, _, _ = self.decoder(conditioned_z)
        
        if self.dynamic_margin:
            
            X_reshaped = torch.reshape(X, (batch_size, self.num_nodes, self.bbx_size+1))
            margin = self.margin_layer(torch.cat([X_reshaped[:, :, 1:], x_bbx], dim=-1))
            margin = self.margin_activation(margin)
            return x_bbx, x_lbl, z_mean, z_logvar, margin

        return x_bbx, x_lbl, z_mean, z_logvar

