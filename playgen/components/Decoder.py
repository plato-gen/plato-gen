import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn

class GCNDecoder(nn.Module):
    """ GCN Decoder module for BoxGCN-Vae. Takes the part presence and adjacency as decoder input.
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 class_size,
                 label_size=1,
                 output_log=False,
                 predict_edges=False,
                 predict_class=False,
                 object_bbox=False,
                 hidden_obj_conditioning=None,
                 hidden_part_conditioning=None,
                 ):
        super(GCNDecoder, self).__init__()

        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.class_size = class_size
        self.label_size = label_size
        self.output_log = output_log
        self.predict_edges = predict_edges
        self.predict_class = predict_class
        self.latent_dims = latent_dims
        
        input_size = latent_dims + class_size
        if object_bbox:
            input_size+=2
        if hidden_obj_conditioning:
            input_size+=(hidden_obj_conditioning-class_size)
        if hidden_part_conditioning:
            input_size+=(hidden_part_conditioning-num_nodes)
            
        self.dense1 = nn.Linear(128,128)
        self.dense_bbx = nn.Linear(128, bbx_size)
        self.dense_bbx_refine = nn.Linear(128, num_nodes*bbx_size)
        self.gconv_bbx_1 = gnn.GCNConv(
            input_size + bbx_size, 128, add_self_loops=True, bias=False)
        self.gconv_bbx_2 = gnn.GCNConv(
            128, 64, add_self_loops=True, bias=False)

        self.dense_lbl = nn.Linear(64, label_size)
        self.dense_edge = nn.Linear(128, num_nodes)
        self.dense_cls = nn.Linear(128,class_size)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Softmax()
        self.act3 = nn.ReLU()

    def forward(self, embedding, E, part_presence):
        
        x_edge = None
        class_pred = None

        x = self.act1(self.gconv_bbx_1(
            torch.cat([embedding, part_presence], axis=-1),
            E))
        x = self.act3(self.dense1(x))
        x_bbx = self.act1(self.gconv_bbx_2(x, E))
        x_bbx = self.act1(self.dense_bbx(x_bbx))           

        x_lbl = self.act1(self.dense_lbl(x))

        if self.predict_edges:
            x_edge = self.act1(self.dense_edge(x))
        
        if self.predict_class:
            class_pred = self.act2(self.dense_cls(x))
 
        return x_bbx, x_lbl, x_edge, class_pred


class Decoder(nn.Module):
    """ Decoder module for BoxGCN-Vae
    """
    def __init__(self,
                 latent_dims,
                 num_nodes,
                 bbx_size,
                 class_size,
                 label_size=1,
                 output_log=False,
                 predict_edges=False,
                 predict_class=False,
                 object_bbox=False,
                 hidden_obj_conditioning=None,
                 hidden_part_conditioning=None,
                 use_gcn=False
                 ):
        super(Decoder, self).__init__()
       
        self.num_nodes = num_nodes
        self.bbx_size = bbx_size
        self.class_size = class_size
        self.label_size = label_size
        self.output_log = output_log
        self.predict_edges = predict_edges
        self.predict_class = predict_class
        self.use_gcn = use_gcn
        
        input_size = latent_dims + num_nodes + class_size
        if object_bbox:
            input_size+=2
        if hidden_obj_conditioning:
            input_size+=(hidden_obj_conditioning-class_size)
        if hidden_part_conditioning:
            input_size+=(hidden_part_conditioning-num_nodes)
            
        self.dense1 = nn.Linear(input_size,128)  
        self.dense2 = nn.Linear(128,128)
        self.dense_bbx = nn.Linear(128,num_nodes*bbx_size)
        self.dense_bbx_refine = nn.Linear(128, num_nodes*bbx_size)
        self.gconv_bbx = gnn.GCNConv(bbx_size, bbx_size, add_self_loops=True, bias=True)

        self.dense_lbl = nn.Linear(128,num_nodes*label_size)
        self.dense_edge = nn.Linear(128,num_nodes*num_nodes)
        self.dense_cls = nn.Linear(128,class_size)
        self.act1 = nn.Sigmoid()
        self.act2 = nn.Softmax()
        self.act3 = nn.ReLU()

    def forward(self, embedding, E=None, training=False, refine_iter=1, labels=None):
        x = self.act1(self.dense1(embedding))
        x = self.act1(self.dense2(x))
        x = self.act1(self.dense2(x))
        
        batch_size = x.shape[0]
        if self.output_log:
            x_bbx = self.act3(self.dense_bbx(x))
        else:
            x_bbx = self.act1(self.dense_bbx(x))
        x_bbx = torch.reshape(x_bbx,[batch_size, self.num_nodes, self.bbx_size])
        
        x_lbl = self.act1(self.dense_lbl(x))
        x_lbl = torch.reshape(x_lbl,[batch_size, self.num_nodes, self.label_size])
        
        x_edge = None
        if self.predict_edges:
            x_edge = self.act1(self.dense_edge(x))
            x_edge = torch.reshape(x_edge,[batch_size, self.num_nodes, self.num_nodes])
        
        class_pred = None
        if self.predict_class:
            class_pred = self.act2(self.dense_cls(x))
        
        if self.use_gcn:
            x_bbx_refined = x_bbx
            for _ in range(refine_iter):
                x_masked = x_bbx_refined * (
                    torch.reshape(labels, x_lbl.shape) 
                    if (labels is not None) else x_lbl)
                if training:
                    x_masked = x_masked * (1.07 - 0.14 * torch.rand(x_masked.shape))
                x_bbx_reshaped = torch.reshape(x_masked, (batch_size*self.num_nodes, self.bbx_size))
                x_correction = self.act1(self.gconv_bbx(x_bbx_reshaped, E))
                x_bbx_refined = torch.reshape(x_bbx_reshaped + x_correction, x_bbx.shape)
            
            
            return x_bbx, x_lbl, x_edge, class_pred, x_bbx_refined
 
        return x_bbx, x_lbl, x_edge, class_pred
