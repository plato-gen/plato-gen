from typing import List
import logging
from src.e2e_pipeline.utils import constants

import numpy as np
import cv2 
import PIL

##For debugging
import pdb 

def binarize(img):
  img = img.convert("L")
  gray = img.point(lambda x: 0 if x<1 else 255, '1')
  return gray

def add_binarized_masks(examples):
  examples["mask_image"] = [
      binarize(image) for image in examples["conditioning_image"]]
  return examples

def binarize_to_layout(img):
  img_np = np.array(img)
  img_np_gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
  mask_np = np.where(img_np_gray == 0, 0, 255).astype(np.uint8)
  mask = PIL.Image.fromarray(mask_np)
  return mask 


def binarize_to_bbx(img):
  img = img.convert("L")
  gray = img.point(lambda x: 0 if x<1 else 255, '1')
  gray_matrix = np.where(np.array(gray))
  x_max, y_max = np.max(gray_matrix, axis=1)
  x_min, y_min = np.min(gray_matrix, axis=1)
  obj_mask = np.zeros(gray.size)
  obj_mask[x_min: x_max+1, y_min: y_max+1] = 255
  obj_mask = np.concatenate((np.expand_dims(obj_mask, axis=-1), np.expand_dims(obj_mask, axis=-1), np.expand_dims(obj_mask, axis=-1)), axis=-1)
  obj_mask = obj_mask.astype(np.uint8)
  return PIL.Image.fromarray(obj_mask)


def add_binarized_obj_mask(examples):
  examples['obj_bbox_mask'] = [
      binarize_to_bbx(image) for image in examples["conditioning_image"]]
  return examples

def get_part_indices(config, obj_name:str, part_names: List[str]) -> np.array:
    part_mapping = constants.ALL_PART_MAPPING.get(obj_name)
    # part_labels = np.zeros(18)
    part_labels = np.zeros(config['layoutgen_model_params']['num_nodes'])
    for part_key in part_names:
        part_idx = part_mapping[part_key]
        part_labels[part_idx] = 1

    return part_labels

def get_adj_matrix(obj_name:str, part_labels:np.array) -> np.matrix:

    all_parts_adj = constants.ALL_ADJ_MAPPING[obj_name]
    condensed_part_mapping = constants.condensed_mapping[obj_name]

    adj_matrix = np.zeros((16, 16))
    present_parts = np.where(part_labels==1)[0].tolist()
    for part_src in present_parts:

        true_part_src = condensed_part_mapping[part_src]
        connected_parts = set(all_parts_adj[true_part_src])
        for part_dst in present_parts:

            true_part_dst = condensed_part_mapping[part_dst]
            if true_part_dst in connected_parts:
                adj_matrix[part_src][part_dst] = 1
                adj_matrix[part_dst][part_src] = 1
            
    return adj_matrix

def get_class_label(config, obj_name:str) -> np.array:
    class_label = np.zeros(config['layoutgen_model_params']['num_classes'])
    class_label[constants.class_dict_inv.get(obj_name)] = 1 
    return class_label


def get_obj_box_dim() -> float:
    return [0.7, 0.7]
    # return [0.5, 0.37]
    # return [0.9, 0.75]
    # return [random.uniform(0.15, 0.85), random.uniform(0.15, 0.85)] 
