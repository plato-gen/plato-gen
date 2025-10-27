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
                 use_gcn=False
                 ):
        super(GCNEncoder, self).__init__()
        
        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.label_size = label_size
        self.num_obj_classes = num_obj_classes
        self.use_gcn = use_gcn 
        
        self.gconv1 = gnn.GCNConv(bbx_size+label_size, hidden1, add_self_loops = False, bias=False, normalize=False)
        self.gconv2 = gnn.GCNConv(hidden1,hidden2, add_self_loops = False, bias=False, normalize=False)
        if  self.use_gcn:
            self.gconv3 = gnn.GCNConv(hidden2,hidden2, add_self_loops = False, bias=False, normalize=False)
        self.dense_boxes = nn.Linear(bbx_size, hidden2)
        self.dense_labels = nn.Linear(label_size,hidden2)
        self.act = nn.ReLU()
        self.dense1 = nn.Linear(16*num_nodes,hidden3)
        self.dense2 = nn.Linear(16*num_nodes+num_obj_classes,hidden3)
        self.dense3 = nn.Linear(hidden3,hidden3)
        
        self.latent = nn.Linear(hidden3,latent_dims)
        # self.fc_mean = nn.Linear(hidden3,latent_dims)
        # self.fc_logvar = nn.Linear(hidden3,latent_dims)

    def forward(self, E, X_data,class_labels):
        
        x = self.gconv1(X_data,E)
        x = self.gconv2(x,E)
        if self.use_gcn:
            x = self.gconv3(x,E)
        
        batch_size = int(x.shape[0]/self.num_nodes)
        x = torch.reshape(x,(batch_size,self.num_nodes*x.shape[-1]))
        
        boxes = X_data[:,1:]
        boxes = torch.reshape(boxes,(batch_size,self.num_nodes,boxes.shape[-1]))                  
        boxes = self.act(self.dense_boxes(boxes)) ##bbx embedding
        
        node_labels = X_data[:,:1]
        node_labels = torch.reshape(node_labels,(batch_size,self.num_nodes,node_labels.shape[-1]))                  
        node_labels = self.act(self.dense_labels(node_labels)) ##node label embedding
        
        mix = torch.add(boxes,node_labels)
        mix = torch.reshape(mix,(batch_size,mix.shape[-2]*mix.shape[-1]))                  
        mix = self.act(self.dense1(mix)) #Concat both
        
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
    """Encoder module for AutoEncoder in BoxGCN-VAE using Graph Attention Networks (GAT)."""

    def __init__(
        self,
        latent_dims,
        num_nodes,
        bbx_size,
        label_size,
        num_obj_classes,
        hidden1,
        hidden2,
        hidden3,
        use_gcn=False,         # kept for interface compatibility
        heads=4,               # number of attention heads
        attn_dropout=0.1       # attention dropout
    ):
        super(GATEncoder, self).__init__()

        self.latent_dims = latent_dims
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.label_size = label_size
        self.num_obj_classes = num_obj_classes
        self.use_gcn = use_gcn  # ignored but kept for compatibility

        # === Graph Attention layers ===
        # concat=True → output dim = out_channels * heads
        self.gat1 = gnn.GATConv(
            in_channels=bbx_size + label_size,
            out_channels=hidden1,
            heads=heads,
            concat=True,
            dropout=attn_dropout,
            add_self_loops=False,
            bias=False
        )

        self.gat2 = gnn.GATConv(
            in_channels=hidden1 * heads,
            out_channels=hidden2,
            heads=heads,
            concat=True,
            dropout=attn_dropout,
            add_self_loops=False,
            bias=False
        )

        # Final attention layer — concat=False collapses heads into one feature vector
        self.gat3 = gnn.GATConv(
            in_channels=hidden2 * heads,
            out_channels=hidden2,
            heads=heads,
            concat=False,
            dropout=attn_dropout,
            add_self_loops=False,
            bias=False
        )

        # === Dense box/label embedding paths ===
        self.dense_boxes = nn.Linear(bbx_size, hidden2)
        self.dense_labels = nn.Linear(label_size, hidden2)

        self.act = nn.ReLU()

        # === Same dense layers as GCNEncoder ===
        self.dense1 = nn.Linear(16 * num_nodes, hidden3)
        self.dense2 = nn.Linear(16 * num_nodes + num_obj_classes, hidden3)
        self.dense3 = nn.Linear(hidden3, hidden3)

        # === Latent projection layers ===
        self.fc_mean = nn.Linear(hidden3, latent_dims)
        self.fc_logvar = nn.Linear(hidden3, latent_dims)

    def forward(self, E, X_data, class_labels):
        """
        Args:
            E: edge index (shape [2, E])
            X_data: node features [B * N, F], where F = bbx_size + label_size
            class_labels: [B, num_obj_classes]
        Returns:
            z_mean, z_logvar
        """

        # === Step 1: GAT Message Passing ===
        x = self.act(self.gat1(X_data, E))
        x = self.act(self.gat2(x, E))
        x = self.act(self.gat3(x, E))

        batch_size = int(x.shape[0] / self.num_nodes)
        x = torch.reshape(x, (batch_size, self.num_nodes * x.shape[-1]))

        # === Step 2: Box and label embeddings ===
        boxes = X_data[:, 1:]
        boxes = torch.reshape(boxes, (batch_size, self.num_nodes, boxes.shape[-1]))
        boxes = self.act(self.dense_boxes(boxes))  # bbx embedding

        node_labels = X_data[:, :1]
        node_labels = torch.reshape(node_labels, (batch_size, self.num_nodes, node_labels.shape[-1]))
        node_labels = self.act(self.dense_labels(node_labels))  # node label embedding

        # Combine embeddings
        mix = torch.add(boxes, node_labels)
        mix = torch.reshape(mix, (batch_size, mix.shape[-2] * mix.shape[-1]))
        mix = self.act(self.dense1(mix))

        # === Step 3: Add class conditioning ===
        class_labels = torch.reshape(class_labels, (batch_size, int(class_labels.shape[-1] / batch_size)))
        x = torch.cat([class_labels, x], dim=-1)
        x = self.act(self.dense2(x))

        # === Step 4: Combine latent paths ===
        x = torch.add(x, mix)
        x = self.act(self.dense3(x))
        x = self.act(self.dense3(x))

        # === Step 5: Mean and logvar ===
        z_mean = self.fc_mean(x)
        z_logvar = self.fc_logvar(x)

        return z_mean, z_logvar


# class GATEncoder(nn.Module):
#     """ GAT module for AutoEncoder to use in Box-VAE. 
#     Args:
#         num_nodes: number of nodes in the encoder.
#     """
#     def __init__(self,
#                  latent_dims,
#                  num_nodes,
#                  bbx_size,
#                  label_size,
#                  num_obj_classes,
#                  hidden1,
#                  hidden2,
#                  hidden3,
#                  ):
#         super(GATEncoder, self).__init__()
        
#         self.latent_dims = latent_dims
#         self.num_nodes = num_nodes
#         self.bbx_size = bbx_size
#         self.label_size = label_size
#         self.num_obj_classes = num_obj_classes
        
#         self.gconv1 = gnn.GATConv(bbx_size+label_size,hidden1, concat=False, add_self_loops = False, bias=False)
#         self.gconv2 = gnn.GATConv(hidden1,hidden2, concat=False, add_self_loops = False, bias=False)
#         self.dense_boxes = nn.Linear(bbx_size, hidden2)
#         self.dense_labels = nn.Linear(label_size,hidden2)
#         self.act = nn.ReLU()
#         self.dense1 = nn.Linear(16*num_nodes,hidden3)
#         self.dense2 = nn.Linear(16*num_nodes+num_obj_classes,hidden3)
#         self.dense3 = nn.Linear(hidden3,hidden3)
        
#         self.latent = nn.Linear(hidden3,latent_dims)

#     def forward(self, E, X_data,class_labels):
        
#         x = self.gconv1(X_data,E)
#         x = self.gconv2(x,E)
        
#         batch_size = int(x.shape[0]/self.num_nodes)
#         x = torch.reshape(x,(batch_size,self.num_nodes*x.shape[-1]))
        
#         boxes = X_data[:,1:]
#         boxes = torch.reshape(boxes,(batch_size,self.num_nodes,boxes.shape[-1]))                  
#         boxes = self.act(self.dense_boxes(boxes))
        
#         node_labels = X_data[:,:1]
#         node_labels = torch.reshape(node_labels,(batch_size,self.num_nodes,node_labels.shape[-1]))                  
#         node_labels = self.act(self.dense_labels(node_labels))
        
#         mix = torch.add(boxes,node_labels)
#         mix = torch.reshape(mix,(batch_size,mix.shape[-2]*mix.shape[-1]))                  
#         mix = self.act(self.dense1(mix))
        
#         class_labels = torch.reshape(class_labels,(batch_size,int(class_labels.shape[-1]/batch_size)))
#         x = torch.cat([class_labels,x],dim=-1)
#         x = self.act(self.dense2(x))
#         # x = torch.add(x,mix)
#         x = self.act(self.dense3(x))
#         x = self.act(self.dense3(x))
        
#         z_mean = self.act(self.latent(x))
#         z_logvar = self.act(self.latent(x))
        
#         return z_mean,z_logvar