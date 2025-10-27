import cv2
import numpy as np
import torch 
import sys

from src.e2e_pipeline.utils import constants

sys.path.append(".")


#For debugging
import pdb 

def sort_bboxes(bbx, key):
    '''
    Sort the bbx matrix based upon the key parameter
    '''
    if key == "area_asc" or key == "area_dsc":
        areas = (bbx[:, 2] - bbx[:, 0]) * (bbx[:, 3] - bbx[:, 1])
        # Get the sorted indices in descending order of area
        if key == "area_dsc":
            sorted_indices = np.argsort(-areas)
        elif key == "area_asc":
            sorted_indices = np.argsort(areas)
    return sorted_indices


def resize_layout(obj_bbx, layout_bbx, num_nodes=16):
    """
    Resize the predicted layouts as the layouts are predicted w.r.t the object bounding boxes.
    """
    layout_bbx = layout_bbx.numpy()
    obj_bbx = torch.cat(((torch.tensor([1.0]).to("cpu")-obj_bbx)*torch.tensor([0.5]).to("cpu"),
                                (torch.tensor([1.0]).to("cpu")+obj_bbx)*torch.tensor([0.5]).to("cpu")),
                            -1)
    
    obj_bbx = np.repeat(
        np.expand_dims(
            (obj_bbx).detach().to("cpu").numpy(), -2
        ),
        num_nodes, -2
    ).squeeze()
    idx_max = np.max(layout_bbx, axis=1)
    idxs = np.where(idx_max != 0)[0]
    layout_bbx[idxs, 0] = obj_bbx[idxs, 0] + (obj_bbx[idxs, 2]-obj_bbx[idxs, 0])*layout_bbx[idxs, 0]
    layout_bbx[idxs, 1] = obj_bbx[idxs, 1] + (obj_bbx[idxs, 3]-obj_bbx[idxs, 1])*layout_bbx[idxs, 1]
    layout_bbx[idxs, 2] = obj_bbx[idxs, 0] + (obj_bbx[idxs, 2]-obj_bbx[idxs, 0])*layout_bbx[idxs, 2]
    layout_bbx[idxs, 3] = obj_bbx[idxs, 1] + (obj_bbx[idxs, 3]-obj_bbx[idxs, 1])*layout_bbx[idxs, 3]
    return layout_bbx



def plot_ellipse(obj_bbx_orig: np.matrix, layout_bbx_orig, rescale=True, scale_to_canvas=True, invert_order: bool=False, sort_type=None) -> np.matrix:
    """Create conditioning image of ellipses

    Args:
        bbx (np.matrix): Layout bounding box
        scale_to_canvas (bool, optional): If bounding boxes need to be scale
            up to the canvas size. Defaults to False.
        invert_order (bool, optional): Should the order of plotting bounding 
            boxes be reversed

    Returns:
        np.matrix: matrix containing the image of layout bounding boxes.
    """
    obj_bbx, layout_bbx = torch.clone(obj_bbx_orig), torch.clone(layout_bbx_orig)
    if rescale:
        bbx = resize_layout(obj_bbx, layout_bbx, num_nodes=18)
    else:
        bbx = layout_bbx 

    if scale_to_canvas:
        bbx = bbx*constants.canvas_size

    if invert_order:
        bbx = bbx[::-1]

    canvas_bbx = np.ones((constants.canvas_size,constants.canvas_size,3), np.uint8) * 255

    canvas = np.zeros(
        (constants.canvas_size, constants.canvas_size, 3), np.uint8) * 255
    obj_x_min = obj_y_min = constants.canvas_size
    obj_x_max = obj_y_max = 0

    if sort_type != None:
        bbx_indices = sort_bboxes(bbx, key=sort_type)
    else:
        bbx_indices = [i for i in range(bbx.shape[0])] 


    bbx_coords_color = []
    # for i, coord in enumerate(bbx):
    for i, bbx_idx in enumerate(bbx_indices):
        # x_minp, y_minp, x_maxp, y_maxp = coord
        x_minp, y_minp, x_maxp, y_maxp = bbx[bbx_idx]

        if x_minp != 0 and y_minp != 0 and x_maxp != 0 and y_maxp != 0:
            obj_x_min = min(obj_x_min, x_minp)
            obj_y_min = min(obj_y_min, y_minp)
            obj_x_max = max(obj_x_max, x_maxp)
            obj_y_max = max(obj_y_max, y_maxp)

            x_center = (x_minp + x_maxp)/2
            y_center = (y_minp + y_maxp)/2
            x_len = max(x_maxp - x_minp, 1)
            y_len = max(y_maxp - y_minp, 1)

            cv2.ellipse(
                canvas,
                (int(x_center), int(y_center)),
                (int(x_len/2), int(y_len/2)),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=constants.colors[bbx_idx].tolist(),
                thickness=-1)
            
            cv2.rectangle(
                canvas_bbx,
                (int(x_minp), int(y_minp)),
                (int(x_maxp) , int(y_maxp) ),
                constants.colors[bbx_idx],
                3)
            bbx_coords_color.append([x_minp, y_minp, x_maxp, y_maxp, constants.colors[bbx_idx]])
    
    return canvas, canvas_bbx, bbx_coords_color 


def plot_obj_bbx(bbx):
    bbx = bbx*constants.canvas_size
    canvas = np.ones((constants.canvas_size,constants.canvas_size,3), np.uint8) * 255
    x_minp, y_minp,x_maxp , y_maxp = bbx
    cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp) , int(y_maxp) ), constants.colors[0], 6)
    return canvas