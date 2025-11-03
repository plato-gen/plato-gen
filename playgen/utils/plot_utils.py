import numpy as np
import torch 
import matplotlib.pyplot as plt
import cv2
import os 
import pdb 

colors = [(1, 0, 0),
          (0.737, 0.561, 0.561),
          (0.255, 0.412, 0.882),
          (0.545, 0.271, 0.0745),
          (0.98, 0.502, 0.447),
          (0.98, 0.643, 0.376),
          (0.18, 0.545, 0.341),
          (0.502, 0, 0.502),
          (0.627, 0.322, 0.176),
          (0.753, 0.753, 0.753),
          (0.529, 0.808, 0.922),
          (0.416, 0.353, 0.804),
          (0.439, 0.502, 0.565),
          (0.784, 0.302, 0.565),
          (0.867, 0.627, 0.867),
          (0, 1, 0.498),
          (0.275, 0.51, 0.706),
          (0.824, 0.706, 0.549),
          (0, 0.502, 0.502),
          (0.847, 0.749, 0.847),
          (1, 0.388, 0.278),
          (0.251, 0.878, 0.816),
          (0.933, 0.51, 0.933),
          (0.961, 0.871, 0.702)]
colors = (np.asarray(colors)*255)
canvas_size = 660


# def plot_bbx(bbx):
#     bbx = bbx*canvas_size
#     canvas = np.ones((canvas_size,canvas_size,3), np.uint8) * 255
#     for i, coord in enumerate(bbx):
#         x_minp, y_minp,x_maxp , y_maxp = coord[:4]
#         if [x_minp, y_minp, x_maxp, y_maxp]!=[0,0,0,0]:
#             cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp) , int(y_maxp) ), colors[i], 3)
#     return canvas

def plot_obj_bbx(bbx):
    bbx = bbx*canvas_size
    canvas = np.ones((canvas_size,canvas_size,3), np.uint8) * 255
    x_minp, y_minp,x_maxp , y_maxp = bbx
    cv2.rectangle(canvas, (int(x_minp), int(y_minp)), (int(x_maxp) , int(y_maxp) ), colors[0], 6)
    return canvas



def scale_bboxes(bbox_coords, obj_boundary_coords):
    data_len, _, _ = bbox_coords.shape
    bbox_coords_scaled = bbox_coords.copy()
    for i in range(data_len):
        try:
            min_x, min_y, max_x, max_y = obj_boundary_coords[i].tolist()
        except:
            min_x, min_y, max_x, max_y = obj_boundary_coords[i].tolist()[0]

        min_x = min_y = min(min_x, min_y)
        max_x = max_y = max(max_x, max_y)
        if bbox_coords.shape[2] == 5:
            labels = np.where(bbox_coords[i, :, :1]==1)[0]
            # Scaling X
            bbox_coords_scaled[i][labels,1] = (bbox_coords[i][labels,1])*(max_x - min_x) + min_x
            bbox_coords_scaled[i][labels,3] = (bbox_coords[i][labels,3])*(max_x - min_x) + min_x
            # #Scaling Y
            bbox_coords_scaled[i][labels,2] = (bbox_coords[i][labels,2])*(max_y - min_y) + min_y
            bbox_coords_scaled[i][labels,4] = (bbox_coords[i][labels,4])*(max_y - min_y) + min_y

        elif bbox_coords.shape[2] == 4:
            # Scaling X
            bbox_coords_scaled[i][:,0] = (bbox_coords[i][:,0])*(max_x - min_x) + min_x
            bbox_coords_scaled[i][:,2] = (bbox_coords[i][:,2])*(max_x - min_x) + min_x
            #Scaling Y
            bbox_coords_scaled[i][:,1] = (bbox_coords[i][:,1])*(max_y - min_y) + min_y
            bbox_coords_scaled[i][:,3] = (bbox_coords[i][:,3])*(max_y - min_y) + min_y
    return bbox_coords_scaled



def plot_bbx(feature_mat_orig, obj_orig):
    feature_mat, obj = np.copy(feature_mat_orig), np.copy(obj_orig)
    scaled_features = scale_bboxes(np.expand_dims(feature_mat, axis=0), np.expand_dims(obj, axis=0))
    # scaled_features = scaled_features * np.reshape(label_true, (scaled_features.shape[0],config.layoutgen_model_params.num_nodes,-1))

    for idx in range(0, scaled_features.shape[0]):
        features = scaled_features[idx, :] #(16*4)
        canvas = np.zeros((660, 660, 3), dtype= np.uint8)
        for part_iter in range(0, features.shape[0]):
            xmin, ymin, xmax, ymax = features[part_iter,0], features[part_iter,1], features[part_iter, 2], features[part_iter, 3]
            if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                continue 
            xmin, xmax = xmin*660, xmax*660
            ymin, ymax = ymin*660, ymax*660
            cv2.rectangle(canvas, (int(xmin), int(ymin)), (int(xmax) , int(ymax)), colors[part_iter], 2)
    return canvas 

save_idx = 0

def plot_pred_gt_layouts(config, gt_feature_mat_orig, pred_feature_mat_orig, label_true, obj_mat_pred, obj_mat_true, class_vec_data):

    gt_feature_mat, pred_feature_mat = gt_feature_mat_orig.copy(), pred_feature_mat_orig.copy()
    def draw_bbx(canvas, xmin, ymin, xmax, ymax, color):
        xmin, ymin, xmax, ymax = xmin*660, ymin*660, xmax*660, ymax*660
        cv2.rectangle(canvas, (int(xmin), int(ymin)), (int(xmax) , int(ymax)), color, 1)
        return canvas

    global save_idx 
    gt_feature_mat = np.reshape(gt_feature_mat, (-1, 16, 5))[:,:,1:]
    pred_scaled_features = scale_bboxes(pred_feature_mat, obj_mat_true.numpy())
    gt_scaled_features = scale_bboxes(gt_feature_mat, obj_mat_true.numpy())
    pred_scaled_features = pred_scaled_features * np.reshape(label_true, (pred_scaled_features.shape[0],config.layoutgen_model_params.num_nodes,-1))
    gt_scaled_features = gt_scaled_features * np.reshape(label_true, (gt_scaled_features.shape[0],config.layoutgen_model_params.num_nodes,-1))

    for idx in range(0, pred_scaled_features.shape[0]):
        pred_features = pred_scaled_features[idx, :] #(16*5)
        gt_features = gt_scaled_features[idx, :]
        class_int = np.argwhere(class_vec_data[idx, :] == 1.)[0][0]
        canvas = np.zeros((660, 660, 3), dtype= np.uint8)
        for part_iter in range(0, pred_features.shape[0]):
            pred_xmin, pred_ymin, pred_xmax, pred_ymax = pred_features[part_iter,0], pred_features[part_iter,1], pred_features[part_iter, 2], pred_features[part_iter, 3]
            gt_xmin, gt_ymin, gt_xmax, gt_ymax = gt_features[part_iter,0], gt_features[part_iter,1], gt_features[part_iter, 2], gt_features[part_iter, 3]

            canvas = draw_bbx(canvas, pred_xmin, pred_ymin, pred_xmax, pred_ymax, color=(0,0,255))
            canvas = draw_bbx(canvas, gt_xmin, gt_ymin, gt_xmax, gt_ymax, color=(255,255,255))

        # cv2.imwrite(f"/home/varghese/MeronymNet-PyTorch/results/palgo_reproduce_v1/class_{class_int}_{idx}.png", canvas)
        cv2.imwrite(os.path.join(config.layoutgen_test_params.output_folder_path, f"class_{class_int}_{save_idx}.png"), canvas)
        save_idx += 1


def plot_layouts(config, feature_mat, label_true, obj_mat, class_vec_data):
    global save_idx 
    scaled_features = scale_bboxes(feature_mat, obj_mat.numpy())
    scaled_features = scaled_features * np.reshape(label_true, (scaled_features.shape[0],config.layoutgen_model_params.num_nodes,-1))

    for idx in range(0, scaled_features.shape[0]):
        features = scaled_features[idx, :] #(16*5)
        class_int = np.argwhere(class_vec_data[idx, :] == 1.)[0][0]
        canvas = np.zeros((660, 660, 3), dtype= np.uint8)
        for part_iter in range(0, features.shape[0]):
            xmin, ymin, xmax, ymax = features[part_iter,0], features[part_iter,1], features[part_iter, 2], features[part_iter, 3]
            if xmin == 0 and ymin == 0 and xmax == 0 and ymax == 0:
                continue 
            xmin, xmax = xmin*660, xmax*660
            ymin, ymax = ymin*660, ymax*660
            cv2.rectangle(canvas, (int(xmin), int(ymin)), (int(xmax) , int(ymax)), (0,255,255), 1)
        # cv2.imwrite(f"/home/varghese/MeronymNet-PyTorch/results/palgo_reproduce_v1/class_{class_int}_{idx}.png", canvas)
        cv2.imwrite(os.path.join(config.layoutgen_test_params.output_folder_path, f"class_{class_int}_{save_idx}.png"), canvas)
        save_idx += 1

        
def xywh_to_xyxy(feature_mat):
    '''
    Input shape: (N, 18, 4) in xywh format
    '''
    feature_mat_xyxy = np.copy(feature_mat)
    feature_mat_xyxy[:,:,0] = feature_mat[:,:,0] - (feature_mat[:,:,2]/2)
    feature_mat_xyxy[:,:,1] = feature_mat[:,:,1] - (feature_mat[:,:,3]/2)
    feature_mat_xyxy[:,:,2] = feature_mat[:,:,0] + (feature_mat[:,:,2]/2)
    feature_mat_xyxy[:,:,3] = feature_mat[:,:,1] + (feature_mat[:,:,3]/2)
    return feature_mat_xyxy
