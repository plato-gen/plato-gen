import numpy as np
import pickle
import random
import copy
import os 

import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader
import torch.utils.data as data_utils

#For debugging
import pdb 


def _batch_generator(node_data, class_labels, obj_data, adj_data, selected_idx_list, batch_size, shuffle=True):
    """
    Batch from individual data points
    """
    train_list =[]
    if obj_data is not None:
        class_data = np.concatenate([class_labels[selected_idx_list], obj_data[selected_idx_list]], axis=-1)
    else:
        class_data = class_labels[selected_idx_list]
    for i, batch in enumerate(zip(copy.deepcopy(node_data[selected_idx_list]),
                                    copy.deepcopy(class_data),
                                    copy.deepcopy(adj_data[selected_idx_list]))):
        if torch.cuda.is_available():
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).cuda().float())
            x_tensor = torch.from_numpy(batch[0]).cuda().float()
            y_tensor = torch.from_numpy(batch[1]).cuda().float()
        else:
            edge_index, _ = dense_to_sparse(torch.from_numpy(batch[2]).float())
            x_tensor = torch.from_numpy(batch[0]).float()
            y_tensor = torch.from_numpy(batch[1]).float()

        data_dict = dict(x=x_tensor, y=y_tensor, edge_index=edge_index)
        train_list.append(Data(**data_dict))
        

    return DataLoader(train_list, batch_size=batch_size, shuffle=shuffle, generator=torch.Generator(device='cuda'))




def load_data(data_path, obj_data_postfix, part_data_post_fix, file_postfix, seed, batch_size, validation=True, \
            use_pose_cond=False, pose_pkl_file=None, shuffle=True, use_pca_centroid=False):   
    """
    Load data from files
    Args:
        data_path (str): base folder path containing data
        obj_data_postfix (str): postfix name for numpy file
        part_data_post_fix (str): postfix name for numpy file
        file_postfix (str): postfix name for numpy file
        seed (int): seed to reproduce validation results only
        batch_size (int): batch_size
        validation (bool, optional): Defaults to True.

    Returns:
        _type_: DataLoader(s)
    """
    base_path = data_path 
    if validation:
        #Loading training data
        outfile = os.path.join(base_path, f"X_train{part_data_post_fix}.np")
        with open(outfile, 'rb') as pickle_file:
            X_train = pickle.load(pickle_file)
        
        outfile = os.path.join(base_path, f"X_train{obj_data_postfix}.np")
        with open(outfile, 'rb') as pickle_file:
            X_obj_train = pickle.load(pickle_file)
        
        outfile = os.path.join(base_path, f"class_v_train{file_postfix}_mask_data.np")
        with open(outfile, 'rb') as pickle_file:
            class_v_train = pickle.load(pickle_file)

        outfile = os.path.join(base_path, f"adj_train{file_postfix}_mask_data.np")
        with open(outfile, 'rb') as pickle_file:
            adj_train = pickle.load(pickle_file)

        ##Loading validation data
        outfile = os.path.join(base_path, f"X_val{part_data_post_fix}.np")
        with open(outfile, 'rb') as pickle_file:
            X_val = pickle.load(pickle_file)

        outfile = os.path.join(base_path, f"X_val{obj_data_postfix}.np")
        with open(outfile, 'rb') as pickle_file:
            X_obj_val = pickle.load(pickle_file)

        outfile = os.path.join(base_path, f"class_v_val{file_postfix}_mask_data.np")
        with open(outfile, 'rb') as pickle_file:
            class_v_val = pickle.load(pickle_file)

        outfile = os.path.join(base_path, f"adj_val{file_postfix}_mask_data.np")
        with open(outfile, 'rb') as pickle_file:
            adj_val = pickle.load(pickle_file)

        # Clip values to [0,1]
        X_train = np.clip(X_train, 0, 1)
        X_val = np.clip(X_val, 0, 1)
        X_obj_train = np.clip(X_obj_train, 0, 1)
        X_obj_val = np.clip(X_obj_val, 0, 1)
    
        # Train indices random (no fixed seed)
        train_idx = np.arange(1, len(X_train))
        val_idx = np.arange(1, len(X_val))

        #NOTE: Comment out this for now
        rng = np.random.default_rng(seed=seed)
        train_idx = rng.permutation(train_idx)
        val_idx = rng.permutation(val_idx)


        batch_train_loader = _batch_generator(
            node_data=X_train,
            class_labels=class_v_train,
            obj_data=X_obj_train,
            adj_data=adj_train,
            selected_idx_list=train_idx,
            batch_size=batch_size,
            shuffle=shuffle)
        
        batch_val_loader = _batch_generator(
            node_data=X_val,
            class_labels=class_v_val,
            obj_data=X_obj_val,
            adj_data=adj_val,
            selected_idx_list=val_idx,
            batch_size=batch_size,
            shuffle=shuffle)
        
        
        return [batch_train_loader, batch_val_loader]
    
    else:
        outfile = os.path.join(base_path, f"X_test{part_data_post_fix}.np")
        with open(outfile, 'rb') as pickle_file:
            X_test = pickle.load(pickle_file)

        outfile = os.path.join(base_path, f"X_test{obj_data_postfix}.np")
        with open(outfile, 'rb') as pickle_file:
            X_obj_test = pickle.load(pickle_file)

        outfile = os.path.join(base_path, f"adj_test{file_postfix}_mask_data.np")
        with open(outfile, 'rb') as pickle_file:
            adj_test = pickle.load(pickle_file)

        outfile = os.path.join(base_path, f"class_v_test{file_postfix}_mask_data.np")
        with open(outfile, 'rb') as pickle_file:
            class_v_test = pickle.load(pickle_file)
        test_idx = np.arange(len(X_test))

        X_test = np.clip(X_test, 0, 1)

        batch_test_loader = _batch_generator(
            node_data=X_test,
            class_labels=class_v_test,
            obj_data=X_obj_test,
            adj_data=adj_test,
            selected_idx_list=test_idx,
            batch_size=batch_size,
            shuffle=shuffle)

        return batch_test_loader
