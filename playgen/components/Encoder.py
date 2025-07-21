import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn


class GCNEncoder(nn.Module):
    """ Encoder module for AutoEncoder in BoxGCN-VAE. 
    Args:
        num_nodes: number of nodes in the encoder.
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 label_size,
                 num_obj_classes,
                 hidden1,
                 hidden2,
                 hidden3,
                 ):
        super(GCNEncoder, self).__init__()
        
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.label_size = label_size
        self.num_obj_classes = num_obj_classes
        
        self.gconv1 = gnn.GCNConv(bbx_size+label_size, hidden1, add_self_loops = False, bias=False, normalize=False)
        self.gconv2 = gnn.GCNConv(hidden1,hidden2, add_self_loops = False, bias=False, normalize=False)
        self.gconv3 = gnn.GCNConv(hidden2,hidden2, add_self_loops = False, bias=False, normalize=False)
        self.dense_boxes = nn.Linear(bbx_size, hidden2)
        self.dense_labels = nn.Linear(label_size,hidden2)
        self.act = nn.ReLU()
        self.dense1 = nn.Linear(hidden2*num_nodes,hidden3)
        self.dense2 = nn.Linear(hidden2*num_nodes+num_obj_classes,hidden3)
        self.dense3 = nn.Linear(hidden3,hidden3)
        
        self.latent = nn.Linear(hidden3,latent_dims)

    def forward(self, E, X_data,class_labels):
        
        x = self.gconv1(X_data,E)
        x = self.gconv2(x,E)
        x = self.gconv3(x,E)
        
        batch_size = int(x.shape[0]/self.num_nodes)
        x = torch.reshape(x,(batch_size,self.num_nodes*x.shape[-1]))
        
        boxes = X_data[:,1:]
        boxes = torch.reshape(boxes,(batch_size,self.num_nodes,boxes.shape[-1]))                  
        boxes = self.act(self.dense_boxes(boxes))
        
        node_labels = X_data[:,:1]
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
        x = self.act(self.dense3(x))
        
        z_mean = self.act(self.latent(x))
        z_logvar = self.act(self.latent(x))
        
        return z_mean,z_logvar
    

class GATEncoder(nn.Module):
    """ GAT module for AutoEncoder to use in Box-VAE. 
    Args:
        num_nodes: number of nodes in the encoder.
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 label_size,
                 num_obj_classes,
                 hidden1,
                 hidden2,
                 hidden3,
                 ):
        super(GATEncoder, self).__init__()
        
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.label_size = label_size
        self.num_obj_classes = num_obj_classes
        
        self.gconv1 = gnn.GATConv(bbx_size+label_size,hidden1, concat=False, add_self_loops = False, bias=False)
        self.gconv2 = gnn.GATConv(hidden1,hidden2, concat=False, add_self_loops = False, bias=False)
        self.dense_boxes = nn.Linear(bbx_size, hidden2)
        self.dense_labels = nn.Linear(label_size,hidden2)
        self.act = nn.ReLU()
        self.dense1 = nn.Linear(hidden2*num_nodes,hidden3)
        self.dense2 = nn.Linear(hidden2*num_nodes+num_obj_classes,hidden3)
        self.dense3 = nn.Linear(hidden3,hidden3)
        
        self.latent = nn.Linear(hidden3,latent_dims)

    def forward(self, E, X_data,class_labels):
        
        x = self.gconv1(X_data,E)
        x = self.gconv2(x,E)
        
        batch_size = int(x.shape[0]/self.num_nodes)
        x = torch.reshape(x,(batch_size,self.num_nodes*x.shape[-1]))
        
        boxes = X_data[:,1:]
        boxes = torch.reshape(boxes,(batch_size,self.num_nodes,boxes.shape[-1]))                  
        boxes = self.act(self.dense_boxes(boxes))
        
        node_labels = X_data[:,:1]
        node_labels = torch.reshape(node_labels,(batch_size,self.num_nodes,node_labels.shape[-1]))                  
        node_labels = self.act(self.dense_labels(node_labels))
        
        mix = torch.add(boxes,node_labels)
        mix = torch.reshape(mix,(batch_size,mix.shape[-2]*mix.shape[-1]))                  
        mix = self.act(self.dense1(mix))
        
        class_labels = torch.reshape(class_labels,(batch_size,int(class_labels.shape[-1]/batch_size)))
        x = torch.cat([class_labels,x],dim=-1)
        x = self.act(self.dense2(x))
        # x = torch.add(x,mix)
        x = self.act(self.dense3(x))
        x = self.act(self.dense3(x))
        
        z_mean = self.act(self.latent(x))
        z_logvar = self.act(self.latent(x))
        
        return z_mean,z_logvar