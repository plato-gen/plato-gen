import sys
import torch
import numpy as np
from torch_geometric.data import DataLoader
import random
import joblib 

from playgen.archs.TwoStageAutoEncoder import TwoStageAutoEncoder
from e2e_pipeline.utils import data_preprocessor
from e2e_pipeline.utils import constants


if torch.cuda.is_available():
    device = torch.device('cuda')
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')

import pdb 



def create_adj_matrices(obj_class, part_indices):
  adj_mat = constants.adj_mat_consol[obj_class]
  parts_int_present = np.argwhere(part_indices == 1.)[:,0].tolist()
  parts_int_absent = [i for i in range(18) if i not in parts_int_present]
  adj_mat[parts_int_absent, :] = 0  # Zero rows
  adj_mat[:, parts_int_absent] = 0  # Zero columns
  return adj_mat




def load_vae(config):
  vae = TwoStageAutoEncoder(config["layoutgen_model_params"]["latent_dims"],
                            config["layoutgen_model_params"]["num_nodes"],
                            config["layoutgen_model_params"]["bbx_size"],
                            config["layoutgen_model_params"]["num_classes"],
                            config["layoutgen_model_params"]["label_shape"],
                            config["layoutgen_model_params"]["hidden1"],
                            config["layoutgen_model_params"]["hidden2"],
                            config["layoutgen_model_params"]["hidden3"],
                            config["layoutgen_model_params"]["dense_hidden1"],
                            config["layoutgen_model_params"]["dense_hidden2"],
                            config["layoutgen_train_params"]["adaptive_margin"],
                            config["layoutgen_train_params"]["output_log"],
                            config["layoutgen_train_params"]["area_encoding"],
                            config["layoutgen_train_params"]["coupling"],
                            config["layoutgen_train_params"]["obj_bbx_conditioning"],
                            config["layoutgen_train_params"]["use_fft_on_bbx"],
                            config["layoutgen_train_params"]["use_gcn_in_decoder"],)
  print(f"{vae = }")
  vae.load_state_dict(torch.load(config["e2e_pipeline"]["layout_gen_weights"]))
  vae = vae.to(device)
  return vae


@torch.no_grad()
def generate_reconstruction_layouts(batch_train_loader, config, model, obj_class, obj_parts, return_latent=True):
  #Generate the filtered dataloader
  obj_class_int = constants.class_dict_inv[obj_class]
  part_labels = data_preprocessor.get_part_indices(config, obj_class, obj_parts)
  part_tensor = torch.unsqueeze(torch.Tensor(part_labels), 0).to(device)


    # --- Prepare filtered list ---
  filtered_data_list = []

  for data in batch_train_loader.dataset:  # Each 'data' is a torch_geometric.data.Data object
      # Move to CPU if needed for numpy operations
      x_np = data.x.detach().cpu().numpy()
      y_np = data.y.detach().cpu().numpy()

      # Extract class and part presence info
      class_vec = y_np[:, :config.layoutgen_model_params.num_classes][0]  # one-hot class vector
      class_int = np.argmax(class_vec)

      part_presence = x_np[:, 0]  # assumes first dim of each part feature = part presence
      part_presence_bin = np.round(part_presence).astype(int)

      # --- Filter condition ---
      if class_int == obj_class_int and np.array_equal(part_presence_bin, np.array(part_labels)):
          filtered_data_list.append(data)

  print(f"Found {len(filtered_data_list)} samples for class '{obj_class}' with given part list")

  # --- Construct new dataloader ---
  if len(filtered_data_list) == 0:
    print("⚠️ No samples matched — check your part list or class name")
    return None
  layouts_ls = []
  obj_bbx_ls = []
  filtered_dataloader = DataLoader(filtered_data_list, batch_size=config.layoutgen_model_params.batch_size)
  for data_iter, train_data in enumerate(filtered_dataloader):
    node_data_true = train_data.x
    label_true = node_data_true[:,:1]
    y_true = train_data.y
    if config.layoutgen_train_params.use_pose_cond == True:
        z_pose_matrix = train_data.pose_cond
    else:
        z_pose_matrix = None 
    class_true = y_true[:, :config.layoutgen_model_params.num_classes]
    X_obj_true = y_true[:, config.layoutgen_model_params.num_classes:]
    X_obj_true_transformed = X_obj_true[:,2:]-X_obj_true[:,:2]
    # X_obj_true_transformed = X_obj_true
    node_data_transformed = torch.cat(
        [node_data_true[:, :-2],
        node_data_true[:, -2:] -  node_data_true[:, -4:-2]], axis=-1)

    adj_true = train_data.edge_index
    batch = train_data.batch
    
    class_true  = torch.flatten(class_true)
    

    output = model(
        adj_true,
        node_data_transformed,
        X_obj_true_transformed,
        label_true,
        class_true, config.layoutgen_train_params.variational, config.layoutgen_train_params.coupling, refine_iter=config.layoutgen_train_params.refine_iter,
        training=False, z_pose_matrix = z_pose_matrix)

    x_bbx_refined = output[8]
    x_bbx_refined_new = torch.cat([x_bbx_refined[:, :, :-2],
                                x_bbx_refined[:, :, -2:] + x_bbx_refined[:, :, -4:-2]], axis=-1).squeeze(0)
    pred_bbx =  (torch.transpose(part_tensor, 0, 1)*x_bbx_refined_new).detach().to("cpu") 
    layouts_ls.append(pred_bbx)

  #Concatenate all the layouts in a tensor
  layouts_ls_tensor = torch.stack(layouts_ls)
  if len(layouts_ls_tensor.size()) == 4:
    layouts_ls_tensor = layouts_ls_tensor.squeeze(0)
  
  return layouts_ls_tensor 
  


def generate_layout(config, model, obj_class, obj_parts, return_latent=True):
  """
  Samples from the VAE latent to generate the layout. 
  """
  class_labels = data_preprocessor.get_class_label(config, obj_class)
  part_labels = data_preprocessor.get_part_indices(config, obj_class, obj_parts)
  obj_box = data_preprocessor.get_obj_box_dim()

  class_tensor = torch.unsqueeze(torch.Tensor(class_labels), 0).to(device)
  part_tensor = torch.unsqueeze(torch.Tensor(part_labels), 0).to(device)
  obj_box_tensor = torch.unsqueeze(torch.Tensor(obj_box), 0).to(device)

  # if z_pose_matrix is not None:
  #   conditioned_pose_latent = torch.cat([z_pose_matrix, z_latent_part], dim=-1)
  # else:
  #   conditioned_pose_latent = z_latent_part 

  ##Object conditioning
  #obj_class = torch.reshape(obj_class, (batch_size, self.num_obj_classes))
  #conditioned_obj_latent = torch.cat([obj_class, z_mean_obj],dim=-1)
  
  ##Part conditioning
  #nodes = torch.reshape(nodes,(batch_size, self.num_nodes))
  #conditioned_obj_latent = torch.cat([nodes, conditioned_obj_latent],dim=-1)
  
  ## object and part representation concat
  #conditioned_z = torch.cat([conditioned_obj_latent, conditioned_pose_latent],dim=-1)


  z_latent_part = torch.normal(torch.zeros([1, 64])).to(device)

  if config.layoutgen_train_params.use_pose_cond:
    z_pose_vector = torch.normal(torch.zeros([1, 10])).to(device)
    conditioned_pose_latent = torch.cat([z_pose_vector, z_latent_part], dim=-1)
  else:
    conditioned_pose_latent = z_pose_vector 

  conditioned_obj_latent = torch.cat([class_tensor, obj_box_tensor],dim=-1).to(device)

  conditioned_obj_latent = torch.cat([part_tensor, conditioned_obj_latent],dim=-1).to(device)
  conditioned_part_latent = torch.cat([conditioned_obj_latent, conditioned_pose_latent],dim=-1).to(device)
  
  output = model.gcn_decoder(conditioned_part_latent)  
  
  node_data_pred_test = output[0]
  pred_bbx =  (torch.transpose(part_tensor, 0, 1)*node_data_pred_test[0]).detach().to("cpu")

  if not return_latent:
    return obj_box_tensor.detach().to("cpu"), pred_bbx 
  else:
    return obj_box_tensor.detach().to("cpu"), pred_bbx, conditioned_part_latent


@torch.no_grad()
def generate_layout_gcn_hierarchical(config, model, obj_class, obj_parts, pose_emb=None, latent_emb=None):
  if latent_emb is not None:
    z_latent_part = latent_emb 
  
  class_labels = data_preprocessor.get_class_label(config, obj_class)
  part_labels = data_preprocessor.get_part_indices(config, obj_class, obj_parts)
  obj_box = data_preprocessor.get_obj_box_dim()

  class_tensor = torch.unsqueeze(torch.Tensor(class_labels), 0).to(device)
  part_tensor = torch.unsqueeze(torch.Tensor(part_labels), 0).to(device)
  obj_box_tensor = torch.unsqueeze(torch.Tensor(obj_box), 0).to(device)

  if config['layoutgen_train_params']['use_pose_cond']:
    if pose_emb is None:
      centroids_dict = joblib.load(config['layoutgen_train_params']['pose_file_path'])
      tuple_part_labels = tuple(part_labels.astype(int).tolist())
      obj_class_int = constants.class_dict_inv[obj_class]
      z_pose_vector_batched = torch.from_numpy(centroids_dict[obj_class_int][tuple_part_labels]).to(device)
      z_pose_vector_batched = z_pose_vector_batched.to(torch.float32)
      # print(f"{z_pose_vector_batched.shape[0] = }")
      z_pose_vector = z_pose_vector_batched[random.randint(0, z_pose_vector_batched.shape[0]-1)]
    else:
      z_pose_vector = pose_emb




    #Create adjacency matrix
    adj_mat = create_adj_matrices(obj_class, part_labels)
    adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
    # Get edge indices from the adjacency matrix
    adj_mat = adj_mat.nonzero(as_tuple=False).t().contiguous()
    adj_mat = adj_mat.to("cuda")

  
    pose_scaled_raw, latent_scaled_raw = model.pose_scale.item()*z_pose_vector, model.latent_scale.item()*z_latent_part
    pose_emb, latent_emb = model.pose_proj(pose_scaled_raw), model.latent_proj(latent_scaled_raw)
    conditioned_obj_latent = torch.cat([class_tensor, obj_box_tensor],dim=-1).to(device)
    conditioned_obj_latent = torch.cat([part_tensor, conditioned_obj_latent],dim=-1).to(device)
    x_bbx, x_lbl, x_edge, class_pred, x_bbx_refined = model.gcn_decoder(conditioned_obj_latent, pose_emb, latent_emb, E=adj_mat, training=False, refine_iter=2)
    
    x_bbx_refined_new = torch.cat([x_bbx_refined[:, :, :-2],
                                  x_bbx_refined[:, :, -2:] + x_bbx_refined[:, :, -4:-2]], axis=-1).squeeze(0)  
    # pred_bbx = (nodes.reshape(nodes.shape[0], nodes.shape[1], 1) * x_bbx_refined_new).detach().to("cpu")
    pred_bbx =  (torch.transpose(part_tensor, 0, 1)*x_bbx_refined_new).detach().to("cpu")

    return obj_box_tensor.detach().to("cpu"), pred_bbx

@torch.no_grad()
def generate_layout_gcn(config, model, obj_class, obj_parts, return_latent=False):
  """
  Samples from the VAE containing the GCN Refinement block
  Args:
      model (_type_): _description_
      obj_class (_type_): _description_
      obj_parts (_type_): _description_
      adj_mat (_type_): _description_
  """
  class_labels = data_preprocessor.get_class_label(config, obj_class)
  part_labels = data_preprocessor.get_part_indices(config, obj_class, obj_parts)
  obj_box = data_preprocessor.get_obj_box_dim()

  class_tensor = torch.unsqueeze(torch.Tensor(class_labels), 0).to(device)
  part_tensor = torch.unsqueeze(torch.Tensor(part_labels), 0).to(device)
  obj_box_tensor = torch.unsqueeze(torch.Tensor(obj_box), 0).to(device)

  # z_latent_part = torch.normal(torch.zeros([1, 64])).to(device) 
  z_latent_part = torch.randn(1, 64).to(device)

  conditioned_obj_latent = torch.cat([class_tensor, obj_box_tensor],dim=-1).to(device)
  conditioned_obj_latent = torch.cat([part_tensor, conditioned_obj_latent],dim=-1).to(device)
  conditioned_part_latent = torch.cat([conditioned_obj_latent, z_latent_part],dim=-1).to(device)

  #Generate the adjacency matrix
  adj_mat = create_adj_matrices(obj_class, part_labels)
  adj_mat = torch.tensor(adj_mat, dtype=torch.float32)
  # Get edge indices from the adjacency matrix
  adj_mat = adj_mat.nonzero(as_tuple=False).t().contiguous()
  adj_mat = adj_mat.to("cuda")
  
  x_bbx, x_lbl, _, _, x_bbx_refined = model.gcn_decoder(conditioned_part_latent, E=adj_mat, refine_iter=1)  
  x_bbx_refined_new = torch.cat([x_bbx_refined[:, :, :-2],
                                  x_bbx_refined[:, :, -2:] + x_bbx_refined[:, :, -4:-2]], axis=-1).squeeze(0)

  # pred_bbx = (nodes.reshape(nodes.shape[0], nodes.shape[1], 1) * x_bbx_refined_new).detach().to("cpu")
  pred_bbx =  (torch.transpose(part_tensor, 0, 1)*x_bbx_refined_new).detach().to("cpu")

  if not return_latent:
    return obj_box_tensor.detach().to("cpu"), pred_bbx
  else:
    return obj_box_tensor.detach().to("cpu"), pred_bbx, conditioned_part_latent
  








