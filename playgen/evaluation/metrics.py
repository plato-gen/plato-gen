import numpy as np
import pandas as pd
import torch

##For debugging
import pdb

def iou(true, pred):
    x1g, y1g, x2g, y2g = np.split(true, 4, -1)
    x1, y1, x2, y2 = np.split(pred, 4, -1)
    
    xA = np.maximum(x1g, x1)
    yA = np.maximum(y1g, y1)
    xB = np.minimum(x2g, x2)
    yB = np.minimum(y2g, y2)
    
    interArea = np.maximum(np.array([0]),(xB - xA))*np.maximum(np.array([0]), (yB - yA))
    boxAArea = np.maximum(np.array([0]), (x2g - x1g))*np.maximum(np.array([0]), (y2g - y1g))
    boxBArea = np.maximum(np.array([0]), (x2 - x1))*np.maximum(np.array([0]),(y2 - y1))
    
    unionArea = boxAArea + boxBArea - interArea
    
    iouk = interArea / unionArea
    
    return iouk
    
def obj_scaling(node, obj):
    node[:, :, 0] = obj[:, :, 0] + (obj[:, :, 2]-obj[:, :, 0])*node[:, :, 0]
    node[:, :, 1] = obj[:, :, 1] + (obj[:, :, 3]-obj[:, :, 1])*node[:, :, 1]
    node[:, :, 2] = obj[:, :, 0] + (obj[:, :, 2]-obj[:, :, 0])*node[:, :, 2]
    node[:, :, 3] = obj[:, :, 1] + (obj[:, :, 3]-obj[:, :, 1])*node[:, :, 3]
    
    return node
    
def get_metrics(node_data_true, X_obj_true, node_data_pred_test, X_obj_pred_test,
                label_true, class_true, num_nodes, num_classes, scaling=True, return_boxes=False):
    
    node_pred = (node_data_pred_test).detach().to("cpu").numpy()
    
    # if scaling:
        # obj_pred = np.repeat(np.expand_dims((X_obj_pred_test).detach().to("cpu").numpy(),-2),num_nodes, -2)
        # node_pred = obj_scaling(node_pred, obj_true)
    
    hasidxs = node_data_true[:, 0].detach().to("cpu").numpy()
    node_true = (node_data_true[:, 1:]).detach().to("cpu").numpy()
    node_true = np.reshape(node_true, node_pred.shape)

    
    if scaling:
        obj_pred = np.repeat(np.expand_dims((X_obj_pred_test).detach().to("cpu").numpy(),-2),num_nodes, -2)
        obj_true = np.reshape(X_obj_true.detach().to("cpu").numpy(), (obj_pred.shape[0],obj_pred.shape[2]))
        obj_true = np.repeat(np.expand_dims(obj_true,-2),num_nodes, -2)
        node_true = obj_scaling(node_true, obj_true)
        node_pred = obj_scaling(node_pred, obj_true)

        
    
    obj_class = np.reshape(class_true.detach().to("cpu").numpy(), (node_data_pred_test.shape[0], num_classes))
    obj_class = np.argmax(obj_class, axis=-1)
    obj_class = np.repeat(np.expand_dims(obj_class, -1), num_nodes, -1)
    label_vals = np.reshape(label_true.detach().to("cpu").numpy(), node_data_pred_test.shape[:2])
    
    node_true_modif = node_true * label_vals.reshape(node_true.shape[0], node_true.shape[1], 1)
    node_pred_modif = node_pred * label_vals.reshape(node_pred.shape[0], node_pred.shape[1], 1)

    # mse_vals_orig = np.mean((node_true-node_pred)**2, axis=-1)*label_vals
    # iou_vals_orig = np.squeeze(iou(node_true, node_pred))*label_vals

    mse_vals = np.mean((node_true_modif-node_pred_modif)**2, axis=-1)
    iou_vals = np.squeeze(iou(node_true_modif, node_pred_modif))

    idxs = np.where(label_vals==1)[1]+1
    label_vals[np.where(label_vals==1)] = idxs

    if return_boxes:
        return node_true, node_pred, pd.DataFrame({"obj_class":obj_class.flatten(),
                         "part_labels":label_vals.flatten(),
                         "IOU":iou_vals.flatten(),
                         "MSE":mse_vals.flatten()})
    
    return pd.DataFrame({"obj_class":obj_class.flatten(),
                         "part_labels":label_vals.flatten(),
                         "IOU":iou_vals.flatten(),
                         "MSE":mse_vals.flatten()})
