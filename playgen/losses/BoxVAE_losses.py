import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj
import random 
import wandb 

##For debugging
import pdb


def _get_smoothening_constants():
    if torch.cuda.is_available():
        smooth_1 = torch.tensor([0.5]).cuda()
        smooth_2 = torch.tensor([1e-08]).cuda()
        zero = torch.tensor([0.0]).cuda()
        one = torch.tensor([1.0]).cuda()
    else:
        smooth_1 = torch.tensor([0.5])
        smooth_2 = torch.tensor([1e-08])
        zero = torch.tensor([0.0])
        one = torch.tensor([1.0])
    
    return smooth_1, smooth_2, zero, one


def kl_loss(z_mean,z_logvar, free_bits=0.03):
    
    # loss = torch.mean(0.5 * torch.sum((torch.square(z_mean) +
    #                                   torch.square(torch.exp(z_logvar)) - 
    #                                   2*(z_logvar) - 1), dim=-1
    #                                  )
    #                  )


    loss = torch.mean(0.5 * torch.sum((torch.square(z_mean) +
                                      torch.exp(z_logvar) - 
                                      z_logvar - 1), dim=-1))
    

    # per-sample, per-dimension KL
    # kl_per_dim = 0.5 * (torch.square(z_mean) + torch.exp(z_logvar) - z_logvar - 1)
    # apply 'free bits' threshold (elementwise clamp)
    # kl_clamped = torch.clamp(kl_per_dim, min=free_bits)
    # sum over dimensions, then average over batch
    # loss = torch.mean(torch.sum(kl_clamped, dim=-1))
    return loss


def adj_loss(pred_edge, true_edge, batch, num_nodes):
    
    true_edge = to_dense_adj(true_edge, batch=batch, max_num_nodes= num_nodes)
    loss = F.binary_cross_entropy(pred_edge, true_edge, reduction='mean')
    
    return loss

def bbox_loss(pred_box, true_box, margin=None, log_output=False, area_encoding=False, has_mse=True, has_obj=False):
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    pred_shape = pred_box.shape
    if area_encoding:
        pred_shape = ([pred_shape[0],pred_shape[1],pred_shape[2]+1])
    
    true_box = torch.reshape(true_box,pred_shape)
    
    if has_obj:
        true_box = true_box[:, 1:]
        pred_box = pred_box[:, 1:]
    
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    if log_output:
        log_pred = pred_box.clone()
        log_true = true_box.clone()
        pred_box = torch.exp(-pred_box)
        true_box = torch.exp(-true_box)
        
    pred_box = torch.multiply(mask, pred_box)
    if area_encoding:
        x1g, y1g, x2g, y2g, areag = torch.tensor_split(true_box, 5, dim=-1)
        x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)
    else:
        x1g, y1g, x2g, y2g = torch.tensor_split(true_box, 4, dim=-1)
        x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)

    xA = torch.maximum(x1g, x1)
    yA = torch.maximum(y1g, y1)
    xB = torch.minimum(x2g, x2)
    yB = torch.minimum(y2g, y2)
    
    w, h = x2g-x1g, y2g-y1g
    
    if torch.is_tensor(margin):
        margin = torch.multiply(mask, margin)
        w_alpha, h_alpha = torch.tensor_split(margin, 2, dim=-1)
        w, h = x2g-x1g, y2g-y1g
        interArea = torch.multiply(torch.maximum(zero,(xB - xA + (w_alpha)*(1-w))), 
                                   torch.maximum(zero, (yB - yA + (h_alpha)*(1-h))))
        boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g + (w_alpha)*(1-w))),
                                  torch.maximum(zero, (y2g - y1g + (h_alpha)*(1-h))))
        boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1 + (w_alpha)*(1-w))),
                                  torch.maximum(zero,(y2 - y1 + (h_alpha)*(1-h))))
    else:
        
        interArea = torch.multiply(torch.maximum(zero,(xB - xA + one)), 
                                   torch.maximum(zero, (yB - yA + one)))
        boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g + one)),
                                  torch.maximum(zero, (y2g - y1g + one)))
        boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1 + one)),
                                  torch.maximum(zero,(y2 - y1 + one)))
    unionArea = boxAArea + boxBArea - interArea + smooth_2
    
    iouk = interArea / unionArea
    iou_loss = -torch.log(iouk + smooth_2)*mask
    iou_loss = torch.mean(iou_loss)
    
    if torch.is_tensor(margin):
        
        iou_loss += torch.mean((one-margin)*mask)
    
    if log_output:
        true_box = log_true*mask#-torch.log(true_box+smooth_2)logtr
        pred_box = log_pred*mask
        iou_loss *= 100
    
    if area_encoding:
        w = pred_box[:,:, 2:3]-pred_box[:, :, 0:1]
        h = pred_box[:, :, 3:4]-pred_box[:, :, 1:2]
        pred_box = torch.cat([pred_box,w*h], dim=-1)
        
    total_loss = iou_loss
    
    if has_mse:
        # Box regression loss
        reg_loss = F.mse_loss(pred_box, true_box, reduction='none')
        reg_loss = torch.mean(reg_loss,dim = -1)
        reg_loss = torch.sum(reg_loss,dim = -1)
        total_non_zero = torch.count_nonzero(reg_loss)
        reg_loss = torch.sum(reg_loss)/(total_non_zero+1)
        total_loss += reg_loss*10
    
    # Pairwise box regression loss
    pair_mse_true = torch.cdist(true_box, true_box)
    pair_mse_pred = torch.cdist(pred_box, pred_box)
    pair_loss = F.mse_loss(pair_mse_true, pair_mse_pred)
    total_non_zero = torch.count_nonzero(torch.sum(pair_loss,dim=-1))
    pair_loss = torch.sum(pair_loss)/(total_non_zero+1)
    
    total_loss += pair_loss
    
    return total_loss

##Disconnection loss for penalizing floating parts
import torch
import torch.nn.functional as F

def compute_disconnection_loss(pred_box, edge_index, batch_vec, margin=0.02):
    """
    Penalizes disconnected parts based on adjacency edges.
    
    Args:
        pred_box: (B*N, 4)
        edge_index: (2, E)
        batch_vec: (B*N,)
        margin: allowed gap between connected parts (0-1 normalized space)
    """
    device = pred_box.device
    src, dst = edge_index.to(device)
    mask = src != dst
    src, dst = src[mask], dst[mask]

    # Compute centers and approximate sizes
    centers = (pred_box[:, :2] + pred_box[:, 2:]) / 2
    widths  = (pred_box[:, 2:3] - pred_box[:, 0:1]).clamp(min=1e-6)
    heights = (pred_box[:, 3:4] - pred_box[:, 1:2]).clamp(min=1e-6)
    sizes   = (widths + heights) / 2

    # Compute per-graph disconnection penalty
    loss_per_graph = []
    for g in batch_vec.unique():
        g_mask = (batch_vec[src] == g) & (batch_vec[dst] == g)
        if not g_mask.any():
            continue
        s, d = src[g_mask], dst[g_mask]
        dist = torch.norm(centers[s] - centers[d], dim=-1)
        target = (sizes[s, 0] + sizes[d, 0]) / 2 + margin
        penalty = F.relu(dist - target)
        loss_per_graph.append(penalty.mean())

    if len(loss_per_graph) == 0:
        return torch.tensor(0.0, device=device)

    return torch.stack(loss_per_graph).mean()



def weighted_bbox_loss(pred_box, true_box, weight=0, margin=None, log_output=False, area_encoding=False, has_obj=False,\
                      adj_mat=None, batch_vec=None, train=True):
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()
    pred_shape = pred_box.shape
    if area_encoding:
        pred_shape = ([pred_shape[0],pred_shape[1],pred_shape[2]+1])
    
    true_box = torch.reshape(true_box,pred_shape)
    
#     if has_obj:
#         true_box = true_box[:, 1:]
#         pred_box = pred_box[:, 1:]

    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    if log_output:
        log_pred = pred_box.clone()
        log_true = true_box.clone()
        pred_box = torch.exp(-pred_box)
        true_box = torch.exp(-true_box)
        
    pred_box = torch.multiply(mask, pred_box)
    if area_encoding:
        x1g, y1g, x2g, y2g, areag = torch.tensor_split(true_box, 5, dim=-1)
        x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)
    else:
        x1g, y1g, x2g, y2g = torch.tensor_split(true_box, 4, dim=-1)
        x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)

    xA = torch.maximum(x1g, x1)
    yA = torch.maximum(y1g, y1)
    xB = torch.minimum(x2g, x2)
    yB = torch.minimum(y2g, y2)
    
    w, h = x2g-x1g, y2g-y1g
    
    if torch.is_tensor(margin):
        margin = torch.multiply(mask, margin)
        w_alpha, h_alpha = torch.tensor_split(margin, 2, dim=-1)
        w, h = x2g-x1g, y2g-y1g
        interArea = torch.multiply(torch.maximum(zero,(xB - xA + (w_alpha)*(1-w))), 
                                   torch.maximum(zero, (yB - yA + (h_alpha)*(1-h))))
        boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g + (w_alpha)*(1-w))),
                                  torch.maximum(zero, (y2g - y1g + (h_alpha)*(1-h))))
        boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1 + (w_alpha)*(1-w))),
                                  torch.maximum(zero,(y2 - y1 + (h_alpha)*(1-h))))
    else: 
        interArea = torch.multiply(torch.maximum(zero,(xB - xA + one)), 
                                   torch.maximum(zero, (yB - yA + one)))
        boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g + one)),
                                  torch.maximum(zero, (y2g - y1g + one)))
        boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1 + one)),
                                  torch.maximum(zero,(y2 - y1 + one)))

    unionArea = boxAArea + boxBArea - interArea + smooth_2
    iouk = interArea / unionArea

    # iouk = torchvision.ops.complete_box_iou_loss(
    #         torch.flatten(true_box, end_dim=-2),
    #         torch.flatten(pred_box, end_dim=-2),
    #         reduction='none'
    #     )

    iou_loss = -torch.log(iouk + smooth_2)*mask
    iou_loss = torch.mean(iou_loss)
    
    if torch.is_tensor(margin):
        
        iou_loss += torch.mean((one-margin)*mask)
    
    if log_output:
        true_box = log_true*mask#-torch.log(true_box+smooth_2)logtr
        pred_box = log_pred*mask
        iou_loss *= 100
    
    if area_encoding:
        w = pred_box[:,:, 2:3]-pred_box[:, :, 0:1]
        h = pred_box[:, :, 3:4]-pred_box[:, :, 1:2]
        pred_box = torch.cat([pred_box,w*h], dim=-1)
        
    # Box regression loss
    reg_loss = F.mse_loss(pred_box, true_box, reduction='none')
    reg_loss = torch.mean(reg_loss,dim = -1)
    reg_loss = torch.sum(reg_loss,dim = -1)
    total_non_zero = torch.count_nonzero(reg_loss)
    reg_loss = torch.sum(reg_loss)/(total_non_zero+1)
    
    # Pairwise box regression loss
    pair_mse_true = torch.cdist(true_box, true_box)
    pair_mse_pred = torch.cdist(pred_box, pred_box)
    pair_loss = F.mse_loss(pair_mse_true, pair_mse_pred)
    total_non_zero = torch.count_nonzero(torch.sum(pair_loss,dim=-1))
    pair_loss = torch.sum(pair_loss)/(total_non_zero+1)
            
    
    return iou_loss*(1+weight) + reg_loss*(1-weight) + pair_loss


def mse_loss(pred_box, true_box):
    pred_shape = pred_box.shape
    true_box = torch.reshape(true_box,pred_shape)
    reg_loss = F.mse_loss(pred_box, true_box, reduction='none')
    reg_loss = torch.mean(reg_loss,dim = -1)
    
    return reg_loss
   
def coarse_bbx_loss(pred_box, true_box):
    pred_shape = pred_box.shape
    true_box = torch.reshape(true_box,pred_shape)
    reg_loss = F.mse_loss(pred_box, true_box, reduction='none')
    reg_loss = torch.mean(reg_loss,dim = -1) #(batch_size, num_nodes)
    reg_loss = torch.sum(reg_loss,dim = -1) #(batch_size, num_nodes)
    total_non_zero = torch.count_nonzero(reg_loss)
    reg_loss = torch.sum(reg_loss)/(total_non_zero+1)
    return reg_loss

def part_intersection_loss(pred_box, true_box, log_output=False, area_encoding=False, mode 
                           =True):
    
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    pred_shape = pred_box.shape
    true_box = torch.reshape(true_box,pred_shape)
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)

    if area_encoding:
        pred_shape = ([pred_shape[0],pred_shape[1],pred_shape[2]+1])
        
    if log_output:
        log_pred = pred_box.clone()
        log_true = true_box.clone()
        pred_box = torch.exp(-pred_box)
        true_box = torch.exp(-true_box)
        
    pred_box = torch.multiply(mask, pred_box)
    batch, num_parts, pars = pred_shape
    
    
    true_rep1 = torch.reshape(true_box.repeat((1,num_parts,1)), (batch, num_parts*num_parts, pars))
    true_rep2 = torch.reshape(true_box.repeat((1,1,num_parts)), (batch, num_parts*num_parts, pars))
    
    pred_rep1 = torch.reshape(pred_box.repeat((1,num_parts,1)), (batch, num_parts*num_parts, pars))
    pred_rep2 = torch.reshape(pred_box.repeat((1,1,num_parts)), (batch, num_parts*num_parts, pars))
    
    if area_encoding:
        xt1_min, yt1_min, xt1_max, yt1_max, _ = torch.tensor_split(true_rep1, 5, dim=-1)
        xt2_min, yt2_min, xt2_max, yt2_max, _ = torch.tensor_split(true_rep2, 5, dim=-1)
        
    else:
        xt1_min, yt1_min, xt1_max, yt1_max = torch.tensor_split(true_rep1, 4, dim=-1)
        xt2_min, yt2_min, xt2_max, yt2_max = torch.tensor_split(true_rep2, 4, dim=-1)
    
    xp1_min, yp1_min, xp1_max, yp1_max = torch.tensor_split(pred_rep1, 4, dim=-1)
    xp2_min, yp2_min, xp2_max, yp2_max = torch.tensor_split(pred_rep2, 4, dim=-1)
    
    xAt = torch.maximum(xt1_min, xt2_min)
    yAt = torch.maximum(yt1_min, yt2_min)
    xBt = torch.minimum(xt1_max, xt2_max)
    yBt = torch.minimum(yt1_max, yt2_max)
    
    xAp = torch.maximum(xp1_min, xp2_min)
    yAp = torch.maximum(yp1_min, yp2_min)
    xBp = torch.minimum(xp1_max, xp2_max)
    yBp = torch.minimum(yp1_max, yp2_max)
    
    inter_area_true = torch.where(torch.multiply(torch.maximum(zero,(xBt - xAt)), 
                                     torch.maximum(zero, (yBt - yAt)))>0, 1, 0)
    
    inter_area_pred = torch.where(torch.multiply(torch.maximum(zero,(xBp - xAp)), 
                                     torch.maximum(zero, (yBp - yAp)))>0, 1, 0)
    
    if binary:
        loss = -torch.mean((torch.eq(inter_area_true, inter_area_pred)).float())/(num_parts*num_parts)
    
    else:
        loss = F.mse_loss(inter_area_true.float(), inter_area_pred.float())
        total_non_zero = torch.count_nonzero(torch.sum(inter_area_true,dim=-1))
        loss = torch.sum(loss)/(total_non_zero+1)
    
    return loss


def intersection_loss(pred_box, true_box, log_output=False, area_encoding=False, binary=True):
    
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    pred_shape = pred_box.shape
    true_box = torch.reshape(true_box,pred_shape)
    batch, num_parts, pars = pred_shape
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    intersection_mask = torch.flatten(torch.unsqueeze(torch.ones((num_parts, num_parts))-torch.eye(num_parts),
                                        0).repeat((batch, 1, 1)), start_dim=1)
    
    if area_encoding:
        pred_shape = ([pred_shape[0],pred_shape[1],pred_shape[2]+1])
        
    if log_output:
        log_pred = pred_box.clone()
        log_true = true_box.clone()
        pred_box = torch.exp(-pred_box)
        true_box = torch.exp(-true_box)
        
    pred_box = torch.multiply(mask, pred_box)
    
    
    true_rep1 = torch.reshape(true_box.repeat((1,num_parts,1)), (batch, num_parts*num_parts, pars))
    true_rep2 = torch.reshape(true_box.repeat((1,1,num_parts)), (batch, num_parts*num_parts, pars))
    
    pred_rep1 = torch.reshape(pred_box.repeat((1,num_parts,1)), (batch, num_parts*num_parts, pars))
    pred_rep2 = torch.reshape(pred_box.repeat((1,1,num_parts)), (batch, num_parts*num_parts, pars))
    
    if area_encoding:
        xt1_min, yt1_min, xt1_max, yt1_max, _ = torch.tensor_split(true_rep1, 5, dim=-1)
        xt2_min, yt2_min, xt2_max, yt2_max, _ = torch.tensor_split(true_rep2, 5, dim=-1)
        
    else:
        xt1_min, yt1_min, xt1_max, yt1_max = torch.tensor_split(true_rep1, 4, dim=-1)
        xt2_min, yt2_min, xt2_max, yt2_max = torch.tensor_split(true_rep2, 4, dim=-1)
    
    xp1_min, yp1_min, xp1_max, yp1_max = torch.tensor_split(pred_rep1, 4, dim=-1)
    xp2_min, yp2_min, xp2_max, yp2_max = torch.tensor_split(pred_rep2, 4, dim=-1)
    
    xAt = torch.maximum(xt1_min, xt2_min)
    yAt = torch.maximum(yt1_min, yt2_min)
    xBt = torch.minimum(xt1_max, xt2_max)
    yBt = torch.minimum(yt1_max, yt2_max)
    
    xAp = torch.maximum(xp1_min, xp2_min)
    yAp = torch.maximum(yp1_min, yp2_min)
    xBp = torch.minimum(xp1_max, xp2_max)
    yBp = torch.minimum(yp1_max, yp2_max)
    
    inter_area_true = torch.where(torch.multiply(torch.maximum(zero,(xBt - xAt)), 
                                     torch.maximum(zero, (yBt - yAt)))>0, 1, 0)
    
    inter_area_pred = torch.where(torch.multiply(torch.maximum(zero,(xBp - xAp)), 
                                     torch.maximum(zero, (yBp - yAp)))>0, 1, 0)
    
    if binary:
        loss = -torch.mean((torch.eq(inter_area_true, inter_area_pred)*intersection_mask).float())/(num_parts*num_parts)
    
    else:
        loss = F.mse_loss(inter_area_true.float(), inter_area_pred.float())*intersection_mask
        total_non_zero = torch.count_nonzero(torch.sum(inter_area_true,dim=-1))
        loss = torch.sum(loss)/(total_non_zero+1)
    
    return loss

def intersection_loss_adjacency_gated(pred_box, true_box, adj_true, log_output=False, area_encoding=False, binary=True):
    
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    pred_shape = pred_box.shape
    true_box = torch.reshape(true_box,pred_shape)
    batch, num_parts, pars = pred_shape
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
#     intersection_mask = torch.flatten(torch.unsqueeze(torch.ones((num_parts, num_parts))-torch.eye(num_parts),
#                                         0).repeat((batch, 1, 1)), start_dim=1)
    
    if area_encoding:
        pred_shape = ([pred_shape[0],pred_shape[1],pred_shape[2]+1])
        
    if log_output:
        log_pred = pred_box.clone()
        log_true = true_box.clone()
        pred_box = torch.exp(-pred_box)
        true_box = torch.exp(-true_box)
        
    pred_box = torch.multiply(mask, pred_box)
    
    
    true_rep1 = torch.reshape(true_box.repeat((1,num_parts,1)), (batch, num_parts*num_parts, pars))
    true_rep2 = torch.reshape(true_box.repeat((1,1,num_parts)), (batch, num_parts*num_parts, pars))
    
    pred_rep1 = torch.reshape(pred_box.repeat((1,num_parts,1)), (batch, num_parts*num_parts, pars))
    pred_rep2 = torch.reshape(pred_box.repeat((1,1,num_parts)), (batch, num_parts*num_parts, pars))
    
    if area_encoding:
        xt1_min, yt1_min, xt1_max, yt1_max, _ = torch.tensor_split(true_rep1, 5, dim=-1)
        xt2_min, yt2_min, xt2_max, yt2_max, _ = torch.tensor_split(true_rep2, 5, dim=-1)
        
    else:
        xt1_min, yt1_min, xt1_max, yt1_max = torch.tensor_split(true_rep1, 4, dim=-1)
        xt2_min, yt2_min, xt2_max, yt2_max = torch.tensor_split(true_rep2, 4, dim=-1)
    
    xp1_min, yp1_min, xp1_max, yp1_max = torch.tensor_split(pred_rep1, 4, dim=-1)
    xp2_min, yp2_min, xp2_max, yp2_max = torch.tensor_split(pred_rep2, 4, dim=-1)
    
    xAt = torch.maximum(xt1_min, xt2_min)
    yAt = torch.maximum(yt1_min, yt2_min)
    xBt = torch.minimum(xt1_max, xt2_max)
    yBt = torch.minimum(yt1_max, yt2_max)
    
    xAp = torch.maximum(xp1_min, xp2_min)
    yAp = torch.maximum(yp1_min, yp2_min)
    xBp = torch.minimum(xp1_max, xp2_max)
    yBp = torch.minimum(yp1_max, yp2_max)
    
    inter_area_true = torch.where(torch.multiply(torch.maximum(zero,(xBt - xAt)), 
                                     torch.maximum(zero, (yBt - yAt)))>0, 1, 0)
    
    inter_area_pred = torch.where(torch.multiply(torch.maximum(zero,(xBp - xAp)), 
                                     torch.maximum(zero, (yBp - yAp)))>0, 1, 0)
    
    if binary:
        loss = -torch.mean((torch.eq(inter_area_true, inter_area_pred)*adj_true).float())/(num_parts*num_parts)
    
    else:
        loss = F.mse_loss(inter_area_true.float(), inter_area_pred.float())*adj_true
        total_non_zero = torch.count_nonzero(torch.sum(inter_area_true,dim=-1))
        loss = torch.sum(loss)/(total_non_zero+1)
    
    return loss

def obj_bbox_loss(pred_box, true_box, weight=0, has_mse=True):
    
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    pred_shape = pred_box.shape
    true_box = torch.reshape(true_box,pred_shape)
    x1g, y1g, x2g, y2g = torch.tensor_split(true_box, 4, dim=-1)
    x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)

    xA = torch.maximum(x1g, x1)
    yA = torch.maximum(y1g, y1)
    xB = torch.minimum(x2g, x2)
    yB = torch.minimum(y2g, y2)
    
    w, h = x2g-x1g, y2g-y1g

    interArea = torch.multiply(torch.maximum(zero,(xB - xA + one)), 
                               torch.maximum(zero, (yB - yA + one)))
    boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g + one)),
                              torch.maximum(zero, (y2g - y1g + one)))
    boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1 + one)),
                              torch.maximum(zero,(y2 - y1 + one)))
    unionArea = boxAArea + boxBArea - interArea + smooth_2
    
    iouk = interArea / unionArea
    iou_loss = -torch.log(iouk + smooth_2)
    iou_loss = torch.mean(iou_loss)
    
    if has_mse:
        # Box regression loss
        reg_loss = (F.mse_loss(pred_box, true_box, reduction='mean') +
                    F.l1_loss(pred_box, true_box, reduction='mean'))
        reg_loss = torch.mean(reg_loss)
    
        return iou_loss*(1+weight) + reg_loss*10*(1-weight)
    
    return iou_loss

def bbox_loss_area_encoding(pred_box, true_box, margin=None, log_output=False):
    
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    pred_shape = pred_box.shape
    pred_shape = ([pred_shape[0],pred_shape[1],pred_shape[2]+1])
    
    true_box = torch.reshape(true_box,pred_shape)
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    if log_output:
        log_pred = pred_box.clone()
        log_true = true_box.clone()
        pred_box = torch.exp(-pred_box)
        true_box = torch.exp(-true_box)
        
    pred_box = torch.multiply(mask, pred_box)

    x1g, y1g, x2g, y2g, areag = torch.tensor_split(true_box, 5, dim=-1)
    x1, y1, x2, y2 = torch.tensor_split(pred_box, 4, dim=-1)

    xA = torch.maximum(x1g, x1)
    yA = torch.maximum(y1g, y1)
    xB = torch.minimum(x2g, x2)
    yB = torch.minimum(y2g, y2)
    
    w, h = x2g-x1g, y2g-y1g
    
    if torch.is_tensor(margin):
        margin = torch.multiply(mask, margin)
        w_alpha, h_alpha = torch.tensor_split(margin, 2, dim=-1)
        w, h = x2g-x1g, y2g-y1g
        interArea = torch.multiply(torch.maximum(zero,(xB - xA + (w_alpha)*(1-w))), 
                                   torch.maximum(zero, (yB - yA + (h_alpha)*(1-h))))
        boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g + (w_alpha)*(1-w))),
                                  torch.maximum(zero, (y2g - y1g + (h_alpha)*(1-h))))
        boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1 + (w_alpha)*(1-w))),
                                  torch.maximum(zero,(y2 - y1 + (h_alpha)*(1-h))))
    else:
        
        interArea = torch.multiply(torch.maximum(zero,(xB - xA + one)), 
                                   torch.maximum(zero, (yB - yA + one)))
        boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g + one)),
                                  torch.maximum(zero, (y2g - y1g + one)))
        boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1 + one)),
                                  torch.maximum(zero,(y2 - y1 + one)))
    unionArea = boxAArea + boxBArea - interArea + smooth_2
    
    iouk = interArea / unionArea
    iou_loss = -torch.log(iouk + smooth_2)*mask
    iou_loss = torch.mean(iou_loss)
    
    if torch.is_tensor(margin):
        
        iou_loss += torch.mean((one-margin)*mask)
    
    area = (pred_box[:,:, 2:3]-pred_box[:, :, 0:1])*(pred_box[:, :, 3:4]-pred_box[:, :, 1:2])
    size = torch.min(pred_box[:, :, 0:2], dim=1, keepdim=True).values-torch.max(pred_box[:, :, 2:4], dim=1, keepdim=True).values
    area = area/(size[:, :, :1]*size[:, :, 1:])
    
        
  
    # Box regression loss
    reg_loss = (F.mse_loss(pred_box, true_box[:, :, :4], reduction='none')
                +F.mse_loss(area, areag, reduction='none')/4)
    reg_loss = torch.mean(reg_loss,dim = -1)
    reg_loss = torch.sum(reg_loss,dim = -1)
    total_non_zero = torch.count_nonzero(reg_loss)
    reg_loss = torch.sum(reg_loss)/(total_non_zero+1)
    
    # Pairwise box regression loss
    pair_mse_true = (torch.cdist(true_box[:, :, :4], true_box[:, :, :4]) + 
                     torch.cdist(areag, areag))
    pair_mse_pred = (torch.cdist(pred_box, pred_box)+ torch.cdist(area, area))
    pair_loss = F.mse_loss(pair_mse_true, pair_mse_pred)
    total_non_zero = torch.count_nonzero(torch.sum(pair_loss,dim=-1))
    pair_loss = torch.sum(pair_loss)/(total_non_zero+1)
    
    return iou_loss + reg_loss + pair_loss


def bbox_loss_hw(pred_box, true_box):
    
    # IOU loss
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    true_box = torch.reshape(true_box,pred_box.shape)
    mask = torch.where(torch.sum(true_box,dim=-1,keepdim=True)!=0,1.0,0.0)
    pred_box = mask*pred_box
    xg, yg, wg, hg = torch.tensor_split(true_box, 4, dim=-1)
    xp, yp, wp, hp = torch.tensor_split(pred_box, 4, dim=-1)
    
    xA = torch.maximum(xg, xp)
    yA = torch.maximum(yg, yp)
    xB = torch.minimum(xg+wg, xp+hp)
    yB = torch.minimum(yg+hg, yp+hp)
    
    interArea = (torch.maximum(zero,
                               (xB - xA + torch.maximum(smooth_1*wg,one))) 
                 * torch.maximum(zero, (yB - yA +torch.maximum(smooth_1*hg,one))))
    boxAArea = (torch.maximum(zero, 
                              (wg + torch.maximum(smooth_1*wg,one))) 
                * torch.maximum(zero,
                                (hg +torch.maximum(smooth_1*hg,one))))
    boxBArea = (torch.maximum(zero, 
                              (wp + torch.maximum(smooth_1*wg,one))) 
                * torch.maximum(zero,
                                (hp + torch.maximum(smooth_1*hg,one))))
    unionArea = boxAArea + boxBArea - interArea + smooth_2
    
    iouk = interArea / unionArea 
    iou_loss = -torch.log(iouk + smooth_2)
    iou_loss = torch.mean(iou_loss)
    
    # Box regression loss
    reg_loss = F.mse_loss(pred_box, true_box, reduction='none')
    reg_loss = torch.mean(reg_loss,dim = -1)
    reg_loss = torch.sum(reg_loss,dim = -1)
    total_non_zero = torch.count_nonzero(reg_loss)
    reg_loss = torch.sum(reg_loss)/(total_non_zero+1)
    
    # Pairwise box regression loss
    pair_mse_true = torch.cdist(true_box, true_box)
    pair_mse_pred = torch.cdist(pred_box, pred_box)
    pair_loss = F.mse_loss(pair_mse_true, pair_mse_pred)
    total_non_zero = torch.count_nonzero(torch.sum(pair_loss,dim=-1))
    pair_loss = torch.sum(pair_loss)/(total_non_zero+1)
    
    return iou_loss + reg_loss + pair_loss
    
def node_loss(pred_nodes, true_nodes, has_obj=False):
    if has_obj:
        pred_shape = pred_nodes.shape
        true_nodes = torch.reshape(true_nodes, (pred_shape[0], pred_shape[1]+1, pred_shape[2]))       
        true_nodes = true_nodes[:, 1:]
    
    else:
        true_nodes = torch.reshape(true_nodes, pred_nodes.shape)
    
    #Find out how many values in pred_nodes are <zero or >one
    # print(f"{torch.count_nonzero(pred_nodes<0) = }, {torch.count_nonzero(pred_nodes>1) = }")
    loss = F.binary_cross_entropy(pred_nodes, true_nodes, reduction='mean')
    
    return loss

def class_loss(pred_class, true_class):
    
    true_class = torch.reshape(true_class, pred_class.shape)
    loss = F.cross_entropy(pred_class,
                           torch.argmax(true_class, dim = -1),
                           reduction='mean')
    
    return loss

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio)

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L
    
def iou(true, pred):
    smooth_1, smooth_2, zero, one = _get_smoothening_constants()

    x1g, y1g, x2g, y2g = torch.tensor_split(true, 4, dim=-1)
    x1, y1, x2, y2 = torch.tensor_split(torch.squeeze(pred), 4, dim=-1)
    
    xA = torch.maximum(x1g, x1)
    yA = torch.maximum(y1g, y1)
    xB = torch.minimum(x2g, x2)
    yB = torch.minimum(y2g, y2)
    
    w, h = x2g-x1g, y2g-y1g
    
    
    interArea = torch.multiply(torch.maximum(zero,(xB - xA)), 
                                   torch.maximum(zero, (yB - yA)))
    boxAArea = torch.multiply(torch.maximum(zero, (x2g - x1g)),
                              torch.maximum(zero, (y2g - y1g)))
    boxBArea = torch.multiply(torch.maximum(zero, (x2 - x1)),
                                  torch.maximum(zero,(y2 - y1)))
    unionArea = boxAArea + boxBArea - interArea
    
    iouk = interArea / unionArea
    
    return iouk   