import os 
import sys
import random  
import cv2
import numpy as np
import pandas as pd
sys.path.append("..")
sys.path.append("/home/varghese/plato-gen/")

import torch
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import DataLoader
import torch.utils.data as data_utils
from torch.utils.tensorboard import SummaryWriter
from losses import BoxVAE_losses as loss
from evaluation import metrics
from utils import plot_utils
from utils import data_utils as data_loading
from archs.DenseAutoencoder import DenseAutoencoder
from archs.DenseAutoencoder import Decoder
from archs.TwoStageAutoEncoder import TwoStageAutoEncoder
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import argparse 
import wandb

#For debugging
import pdb 

class_dict = {
    0: 'cow',
    1: 'sheep',
    2: 'bird',
    3: 'person',
    4: 'cat',
    5: 'dog',
    6: 'horse',
    7: 'aeroplane',
    8: 'motorbike',
    9: 'bicycle',
    10: 'pottedplant',
}

###################################################Global variables##########################################################
if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device('cpu')
#########################################################Arguments###########################################################
parser = argparse.ArgumentParser(description="Palgo train and test script")
parser.add_argument("--train", action="store_true", help="Enable training mode")
parser.add_argument("--test", action="store_true", help="Enable testing mode")
args = parser.parse_args()

def train():
    """
    Training and validation Code
    """
    reconstruction_loss_arr = []
    kl_loss_obj_arr = []
    kl_loss_part_arr = []
    bbox_loss_arr = []
    refined_bbox_loss_arr = []
    adj_loss_arr = []
    node_loss_arr = []

    reconstruction_loss_val_arr = []
    kl_loss_val_arr = []
    bbox_loss_val_arr = []
    refined_bbox_loss_val_arr = []
    adj_loss_val_arr = []
    node_loss_val_arr = []

    bbox_loss_threshold = 3.0
    vae = TwoStageAutoEncoder(config.layoutgen_model_params.latent_dims,
                            config.layoutgen_model_params.num_nodes,
                            config.layoutgen_model_params.bbx_size,
                            config.layoutgen_model_params.num_classes,
                            config.layoutgen_model_params.label_shape,
                            config.layoutgen_model_params.hidden1,
                            config.layoutgen_model_params.hidden2,
                            config.layoutgen_model_params.hidden3,
                            config.layoutgen_model_params.dense_hidden1,
                            config.layoutgen_model_params.dense_hidden2,
                            config.layoutgen_train_params.adaptive_margin,
                            config.layoutgen_train_params.output_log,
                            config.layoutgen_train_params.area_encoding,
                            config.layoutgen_train_params.coupling,
                            config.layoutgen_train_params.obj_bbx_conditioning,
                            config.layoutgen_train_params.use_fft_on_bbx,
                            config.layoutgen_train_params.use_gcn_in_decoder,
                            )
    vae.to(device)
    optimizer = optim.Adam(vae.parameters(), lr=config.layoutgen_train_params.learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,300], gamma=0.75)
    model_path = ('./playgen/runs/model/'+config.layoutgen_train_params.run_prefix+'/Obj-Box-'
                            +str(config.layoutgen_train_params.learning_rate)
                            +'-batch-'+str(config.layoutgen_model_params.batch_size)
                            +'-h1-'+str(config.layoutgen_model_params.hidden1)
                            +'-h2-'+str(config.layoutgen_model_params.hidden2)
                            +'-h3-'+str(config.layoutgen_model_params.hidden3)+'-test')
    summary_path = ('./playgen/runs/summary/'+config.layoutgen_train_params.run_prefix+'/Obj-Box-'
                            +str(config.layoutgen_train_params.learning_rate)
                            +'-batch-'+str(config.layoutgen_model_params.batch_size)
                            +'-h1-'+str(config.layoutgen_model_params.hidden1)
                            +'-h2-'+str(config.layoutgen_model_params.hidden2)
                            +'-h3-'+str(config.layoutgen_model_params.hidden3)+'-test')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    # writer = SummaryWriter(summary_path) -> REMOVED
    icoef = 0
    for epoch in range(config.layoutgen_train_params.nb_epochs):   
        batch_loss = torch.tensor([0.0])
        batch_kl_loss_part = torch.tensor([0.0])
        batch_kl_loss_obj = torch.tensor([0.0])
        batch_bbox_loss = torch.tensor([0.0])
        batch_refined_bbox_loss = torch.tensor([0.0])
        batch_obj_bbox_loss = torch.tensor([0.0])
        batch_node_loss = torch.tensor([0.0])
        batch_coarse_bbox_loss = torch.tensor([0.0])
        IOU_weight_delta = torch.tensor([(1+epoch)/config.layoutgen_train_params.nb_epochs])
        images = []
        
        vae.train()
        i=0
        for train_iter, train_data in enumerate(batch_train_loader):
            print(f"Inside train loop...")
            node_data_true = train_data.x
            label_true = node_data_true[:,:1]
            y_true = train_data.y
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
            

            for param in vae.parameters():
                param.grad=None
            
            output = vae(
                adj_true,
                node_data_transformed,
                X_obj_true_transformed,
                label_true,
                class_true, config.layoutgen_train_params.variational, config.layoutgen_train_params.coupling, refine_iter=config.layoutgen_train_params.refine_iter,
                training=False)
                # training=(epoch>100))

            node_data_pred = output[0]
            node_data_pred_new = torch.cat(
                [node_data_pred[:, :, :-2],
                node_data_pred[:, :, -2:] + node_data_pred[:, :, -4:-2]], axis=-1)
            X_obj_pred = output[1]
            X_obj_pred = torch.cat(((torch.tensor([1.0])-X_obj_pred)*torch.tensor([0.5]),
                                    (torch.tensor([1.0])+X_obj_pred)*torch.tensor([0.5])),
                                -1)
            label_pred = output[2]
            z_mean_part = output[3]
            z_logvar_part = output[4]
            margin = output[5]
            z_mean_obj = output[6]
            z_logvar_obj = output[7]

 
            if config.layoutgen_train_params.use_gcn_in_decoder:
                node_data_pred_refined = output[8]
                node_data_pred_refined_new = torch.cat(
                    [node_data_pred_refined[:, :, :-2],
                    node_data_pred_refined[:, :, -2:] + node_data_pred_refined[:, :, -4:-2]], axis=-1)

            if config.layoutgen_train_params.variational:
                kl_loss_part = loss.kl_loss(z_mean_part, z_logvar_part)
            else:
                kl_loss_part = 0.0 
            coarse_bbox_loss = loss.coarse_bbx_loss(pred_box=node_data_pred_new, true_box=node_data_true[:,1:])
            # bbox_loss = loss.weighted_bbox_loss(pred_box=node_data_pred_new, true_box=node_data_true[:,1:], weight=IOU_weight_delta, margin=margin)
            node_loss = loss.node_loss(label_pred,label_true) #Loss: BCE loss on the nodes
            

            if config.layoutgen_train_params.use_gcn_in_decoder:
                refined_bbox_loss = loss.weighted_bbox_loss(
                    pred_box=node_data_pred_refined_new,
                    true_box=node_data_true[:, 1:],
                    weight=IOU_weight_delta,
                    margin=margin,
                    adj_mat=adj_true, 
                    batch_vec=train_data.batch,
                    train=True
                )  # MSE loss + Pairwise MSE loss + Dynamic margin IOU loss
                reconstruction_loss = (coarse_bbox_loss + refined_bbox_loss + node_loss) * config.layoutgen_model_params.num_nodes # * epoch / nb_epochs

            
            else:
                refined_bbox_loss = torch.tensor([0.0])
                reconstruction_loss = (bbox_loss + node_loss) * config.layoutgen_model_params.num_nodes # * epoch / nb_epochs

            
            kl_weight = klw[icoef]

            if config.layoutgen_train_params.variational and (kl_weight>0):
                reconstruction_loss += (kl_loss_part*kl_weight)   
            
            # if epoch >200:
            #     reconstruction_loss += bbox_loss

            reconstruction_loss.backward()
            
            optimizer.step()
            
            i+=1
        
            batch_loss += reconstruction_loss
            batch_kl_loss_part += kl_loss_part
            batch_coarse_bbox_loss += coarse_bbox_loss
            # batch_kl_loss_obj += kl_loss_obj
            # batch_bbox_loss += bbox_loss
            batch_refined_bbox_loss += refined_bbox_loss
            # batch_obj_bbox_loss += obj_bbox_loss
            batch_node_loss += node_loss
    

            
        scheduler.step()
        global_step = epoch*len(batch_train_loader)+i
        # Log to W&B
        wandb.log({
            "Loss/train/reconstruction_loss": batch_loss.item() / (i+1),
            "Loss/train/kl_loss_part": batch_kl_loss_part.item() / (i+1),
            "Loss/train/coarse_bbox_loss": batch_coarse_bbox_loss.item() / (i+1),
            "Loss/train/refined_bbox_loss": batch_refined_bbox_loss.item() / (i+1),
            "Loss/train/node_loss": batch_node_loss.item() / (i+1),
            "kl_weight": kl_weight,
            "epoch": epoch
        }, step=epoch)
        # Optional: To log to Tensorboard
        # writer.add_scalar("Loss/train/reconstruction_loss", batch_loss.item()/(i+1), global_step)
        # writer.add_scalar("Loss/train/kl_loss_part", batch_kl_loss_part.item()/(i+1), global_step)
        # writer.add_scalar("Loss/train/refined_bbox_loss", batch_refined_bbox_loss.item()/(i+1), global_step)
        # writer.add_scalar("Loss/train/node_loss", batch_node_loss.item()/(i+1), global_step)
        # writer.add_scalar("Loss/train/kl_loss", kl_loss_part * kl_weight)


        image_shape = [config.layoutgen_model_params.num_nodes, config.layoutgen_model_params.bbx_size]
        train_rand_idxs_ls = list(range(0, int(node_data_true.shape[0]/config.layoutgen_model_params.num_nodes-1), 5))
        
        # Log validation images to W&B
        if epoch % 10 == 0:
            for rand_iter, rand_idx in enumerate(train_rand_idxs_ls):
                start_idx = rand_idx * config.layoutgen_model_params.num_nodes
                start_idx_class = rand_idx * config.layoutgen_model_params.num_classes
                class_onehot = class_true[start_idx_class: start_idx_class + config.layoutgen_model_params.num_classes]
                class_name = class_dict[torch.argwhere(class_onehot == 1.)[0][0].item()]

                image_gt = plot_utils.plot_bbx(np.reshape((node_data_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes,1:5]*label_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes]).detach().to("cpu").numpy(),
                                            image_shape), [0.0, 0.0, 0.8, 0.8])
                image_pred = plot_utils.plot_bbx(node_data_pred_new[rand_idx].detach().to("cpu").numpy()*label_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes].detach().to("cpu").numpy(), \
                                            [0.0, 0.0, 0.8, 0.8])
                
                #Save format: input_epoch_{}_index_{}
                # f"{i:03d}.jpg"
                # cv2.imwrite(os.path.join(config.layoutgen_train_params.tmp_save_path, f"train_input_epoch_{epoch}_index_{rand_iter:03d}.jpg"), image_gt)
                # cv2.imwrite(os.path.join(config.layoutgen_train_params.tmp_save_path, f"train_pred_epoch_{epoch}_index_{rand_iter:03d}.jpg"), image_pred)
                
                wandb.log({f"train/images/{class_name}_input_{rand_iter}": wandb.Image(image_gt)}, step=epoch)
                wandb.log({f"train/images/{class_name}_pred_{rand_iter}": wandb.Image(image_pred)}, step=epoch)
            
                if config.layoutgen_train_params.use_gcn_in_decoder:
                    image_refined = plot_utils.plot_bbx((node_data_pred_refined_new[rand_idx].detach().to("cpu").numpy()*label_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes].detach().to("cpu").numpy()), \
                        [0.0, 0.0, 0.8, 0.8])
                    # cv2.imwrite(os.path.join(config.layoutgen_train_params.tmp_save_path, f"train_refined_epoch_{epoch}_index_{rand_iter:03d}.jpg"), image_refined)
                    wandb.log({f"train/images/{class_name}_refined_{rand_iter}": wandb.Image(image_refined)}, step=epoch)

                # REMOVED
                # writer.add_image('train/images/refined', image, global_step, dataformats='HWC')
                # cv2.imwrite(f"/ssd_scratch/varghese/training_runs/train_image_pred_refined.png", image)
        
        reconstruction_loss_arr.append(batch_loss.detach().item()/(i+1))
        kl_loss_obj_arr.append(batch_kl_loss_obj.detach().item()/(i+1))
        kl_loss_part_arr.append(batch_kl_loss_part.detach().item()/(i+1))
        bbox_loss_arr.append(batch_bbox_loss.detach().item()/(i+1))
        refined_bbox_loss_arr.append(batch_refined_bbox_loss.detach().item()/(i+1))
        node_loss_arr.append(batch_node_loss.detach().item()/(i+1))
        
        print('[%d, %5d] Train loss: %.3f' %
                (epoch + 1, i + 1, batch_loss/(i+1) ))
    
    
        batch_loss = torch.tensor([0.0])
        batch_kl_loss_part = torch.tensor([0.0])
        batch_kl_loss_obj = torch.tensor([0.0])
        batch_bbox_loss = torch.tensor([0.0])
        batch_refined_bbox_loss = torch.tensor([0.0])
        batch_obj_bbox_loss = torch.tensor([0.0])
        batch_node_loss = torch.tensor([0.0])
        batch_coarse_bbox_loss = torch.tensor([0.0])
        images = []
        vae.eval()
        for i, val_data in enumerate(batch_val_loader, 0):
            node_data_true = val_data.x
            label_true = node_data_true[:,:1]
            y_true = val_data.y
            class_true = torch.flatten(y_true[:, :config.layoutgen_model_params.num_classes])
            X_obj_true = y_true[:, config.layoutgen_model_params.num_classes:]
            # X_obj_true_transformed = X_obj_true
            X_obj_true_transformed = X_obj_true[:,2:]-X_obj_true[:,:2]
            node_data_transformed = torch.cat(
                [node_data_true[:, :-2],
                node_data_true[:, -2:] -  node_data_true[:, -4:-2]], axis=-1)
            adj_true = val_data.edge_index
            batch = val_data.batch
            
            class_true  = torch.flatten(class_true)
            
            output = vae(adj_true, node_data_transformed, X_obj_true_transformed, label_true , class_true, config.layoutgen_train_params.variational, config.layoutgen_train_params.coupling, \
                         refine_iter=config.layoutgen_train_params.refine_iter, training=False)
            
            node_data_pred = output[0]
            node_data_pred_new = node_data_pred
            node_data_pred_new = torch.cat(
                [node_data_pred[:, :, :-2],
                node_data_pred[:, :, -2:] + node_data_pred[:, :, -4:-2]], axis=-1)

            X_obj_pred = output[1]
            X_obj_pred = torch.cat(((torch.tensor([1.0])-X_obj_pred)*torch.tensor([0.5]),
                                    (torch.tensor([1.0])+X_obj_pred)*torch.tensor([0.5])),
                                -1)
            label_pred = output[2]
            z_mean_part = output[3]
            z_logvar_part = output[4]
            margin = output[5]
            z_mean_obj = output[6]
            z_logvar_obj = output[7]

            if config.layoutgen_train_params.use_gcn_in_decoder:
                node_data_pred_refined = output[8]
                node_data_pred_refined_new = node_data_pred_refined
                node_data_pred_refined_new = torch.cat(
                    [node_data_pred_refined_new[:, :, :-2],
                    node_data_pred_refined_new[:, :, -2:] + node_data_pred_refined_new[:, :, -4:-2]], axis=-1)
            
            
            # obj_bbox_loss = loss.obj_bbox_loss(pred_box=X_obj_pred, true_box=X_obj_true, weight=IOU_weight_delta, has_mse=False)
            # kl_loss_obj = loss.kl_loss(z_mean_obj, z_logvar_obj)
            coarse_bbox_loss = loss.coarse_bbx_loss(pred_box=node_data_pred_new, true_box=node_data_true[:,1:])
            if config.layoutgen_train_params.variational:
                kl_loss_part = loss.kl_loss(z_mean_part, z_logvar_part)
            else:
                kl_loss_part = 0.0
            # bbox_loss = loss.weighted_bbox_loss(pred_box=node_data_pred_new, true_box=node_data_true[:,1:], weight=IOU_weight_delta, margin=margin)
            # refined_bbox_loss = loss.weighted_bbox_loss(pred_box=node_data_pred_refined_new, true_box=node_data_true[:,1:], weight=IOU_weight_delta, margin=margin)
            node_loss = loss.node_loss(label_pred,label_true)
            #Refinement loss
            if config.layoutgen_train_params.use_gcn_in_decoder:
                refined_bbox_loss = loss.weighted_bbox_loss(
                    pred_box=node_data_pred_refined_new,
                    true_box=node_data_true[:, 1:],
                    weight=IOU_weight_delta,
                    margin=margin,
                    adj_mat=adj_true,
                    batch_vec=val_data.batch,
                    train=False 
                )
                reconstruction_loss = (coarse_bbox_loss + refined_bbox_loss + node_loss) * config.layoutgen_model_params.num_nodes # * epoch / nb_epochs
                # reconstruction_loss = (coarse_bbox_loss + refined_bbox_loss + node_loss) # * epoch / nb_epochs

            
            else:
                refined_bbox_loss = torch.tensor([0.0])
                reconstruction_loss = (bbox_loss + node_loss) * config.layoutgen_model_params.num_nodes # * epoch / nb_epochs
                # reconstruction_loss = (bbox_loss + node_loss)# * epoch / nb_epochs


            kl_weight = klw[icoef]


            if config.layoutgen_train_params.variational and (kl_weight>0):
                reconstruction_loss += (kl_loss_part * kl_weight)
            
                
            batch_loss += reconstruction_loss
            batch_kl_loss_part += kl_loss_part
            batch_coarse_bbox_loss += coarse_bbox_loss
            # batch_kl_loss_obj += kl_loss_obj
            # batch_bbox_loss += bbox_loss
            batch_refined_bbox_loss += refined_bbox_loss
            # batch_obj_bbox_loss += obj_bbox_loss
            batch_node_loss += node_loss



        #Print validation loss
        print('[%d, %5d] Validation loss: %.3f' % (epoch + 1, i + 1, batch_loss/(i+1) ))

        ##Write validation images
        image_shape = [config.layoutgen_model_params.num_nodes, config.layoutgen_model_params.bbx_size]
        val_rand_idxs_ls = list(range(0, int(node_data_true.shape[0]/config.layoutgen_model_params.num_nodes), 5))
        # print(f"{val_rand_idxs_ls = }")
        # rand_idx = random.randint(0, node_data_true.shape[0]/config.layoutgen_model_params.num_nodes-1)
        
        # Log validation images to W&B
        if epoch % 10 == 0:
            for rand_iter, rand_idx in enumerate(val_rand_idxs_ls):
                start_idx = rand_idx * config.layoutgen_model_params.num_nodes
                start_idx_class = rand_idx * config.layoutgen_model_params.num_classes
                class_onehot = class_true[start_idx_class: start_idx_class + config.layoutgen_model_params.num_classes]
                class_name = class_dict[torch.argwhere(class_onehot == 1.)[0][0].item()]
                val_image_gt = plot_utils.plot_bbx(np.reshape((node_data_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes,1:5]*label_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes]).detach().to("cpu").numpy(),
                                            image_shape), [0.0, 0.0, 0.8, 0.8])
                val_image_pred = plot_utils.plot_bbx(node_data_pred_new[rand_idx].detach().to("cpu").numpy()*label_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes].detach().to("cpu").numpy(), \
                                            [0.0, 0.0, 0.8, 0.8])
                wandb.log({f"val/images/{class_name}_input_{rand_iter}": wandb.Image(val_image_gt)}, step=epoch)
                wandb.log({f"val/images/{class_name}_pred_{rand_iter}": wandb.Image(val_image_pred),}, step=epoch)

            # REMOVED
            # cv2.imwrite(f"/ssd_scratch/varghese/training_runs/val_image_gt.png", image)
            # cv2.imwrite(f"/ssd_scratch/varghese/training_runs/val_image_pred.png", image)

                if config.layoutgen_train_params.use_gcn_in_decoder:
                    val_image_refined = plot_utils.plot_bbx((node_data_pred_refined_new[rand_idx].detach().to("cpu").numpy()*label_true[start_idx:start_idx + config.layoutgen_model_params.num_nodes].detach().to("cpu").numpy()), \
                        [0.0, 0.0, 0.8, 0.8])
                    wandb.log({f"val/images/{class_name}_refined_{rand_iter}": wandb.Image(val_image_refined)}, step=epoch)
                # REMOVED
                # cv2.imwrite(f"/ssd_scratch/varghese/training_runs/val_image_pred_refined.png", image)

        reconstruction_loss_val_arr.append(batch_loss.detach().item()/(i+1))
        bbox_loss_val_arr.append(batch_bbox_loss.detach().item()/(i+1))
        node_loss_val_arr.append(batch_node_loss.detach().item()/(i+1))
        
        # Log validation metrics to W&B
        wandb.log({
            "Loss/val/reconstruction_loss": batch_loss.detach().item()/(i+1),
            "Loss/val/coarse_bbox_loss": batch_coarse_bbox_loss.item()/(i+1),
            "Loss/val/kl_loss_part": batch_kl_loss_part.item()/(i+1),
            "Loss/val/refined_bbox_loss": batch_refined_bbox_loss.item()/(i+1),
            "Loss/val/node_loss": batch_node_loss.item()/(i+1),
            "epoch": epoch
        }, step=epoch)
        
        # REMOVED
        # writer.add_scalar("Loss/val/reconstruction_loss", batch_loss.detach()/(i+1), global_step)
        # writer.add_scalar("Loss/train/kl_loss_part", batch_kl_loss_part.item()/(i+1), global_step)
        # writer.add_scalar("Loss/val/refined_bbox_loss", batch_refined_bbox_loss.item()/(i+1), global_step)
        # writer.add_scalar("Loss/val/node_loss", batch_node_loss.item()/(i+1), global_step)

        if epoch%100 == 0:
            # Save model as a W&B artifact
            torch.save(vae.state_dict(), model_path + f'/model_weights_{epoch}.pth')
            
        if ((kl_loss_part_arr[-1]>0.5) and (abs(bbox_loss_arr[-1] - bbox_loss_val_arr[-1]) < 0.07) and \
            (bbox_loss_arr[-1]<bbox_loss_threshold) and (epoch>300) and config.layoutgen_train_params.variational):
            icoef = icoef + 1
            bbox_loss_threshold*=0.9
            print(f"{kl_loss_part_arr[-1] = }")
            print(f"{refined_bbox_loss_arr[-1] = }")

    torch.save(vae.state_dict(),model_path + '/model_weights.pth')
    
    # REMOVED
    # writer.flush()
    # writer.close()
    
    print('Finished Training')



def test():
    """
    Testing Code 
    """
    model_path = ('/home/varghese/palgo/src/layout_generation/runs/model/'+config.layoutgen_train_params.run_prefix+'/Obj-Box-'
                            +str(config.layoutgen_train_params.learning_rate)
                            +'-batch-'+str(config.layoutgen_model_params.batch_size)
                            +'-h1-'+str(config.layoutgen_model_params.hidden1)
                            +'-h2-'+str(config.layoutgen_model_params.hidden2)
                            +'-h3-'+str(config.layoutgen_model_params.hidden3)+'-test')
    summary_path = ('/home/varghese/palgo/src/layout_generation/runs/summary/'+config.layoutgen_train_params.run_prefix+'/Obj-Box-'
                            +str(config.layoutgen_train_params.learning_rate)
                            +'-batch-'+str(config.layoutgen_model_params.batch_size)
                            +'-h1-'+str(config.layoutgen_model_params.hidden1)
                            +'-h2-'+str(config.layoutgen_model_params.hidden2)
                            +'-h3-'+str(config.layoutgen_model_params.hidden3)+'-test')


    write_tensorboard = False
    # if write_tensorboard:
        # writer = SummaryWriter(summary_path)

    vae = TwoStageAutoEncoder(config.layoutgen_model_params.latent_dims,
                            config.layoutgen_model_params.num_nodes,
                            config.layoutgen_model_params.bbx_size,
                            config.layoutgen_model_params.num_classes,
                            config.layoutgen_model_params.label_shape,
                            config.layoutgen_model_params.hidden1,
                            config.layoutgen_model_params.hidden2,
                            config.layoutgen_model_params.hidden3,
                            config.layoutgen_model_params.dense_hidden1,
                            config.layoutgen_model_params.dense_hidden2,
                            config.layoutgen_train_params.adaptive_margin,
                            config.layoutgen_train_params.output_log,
                            config.layoutgen_train_params.area_encoding,
                            config.layoutgen_train_params.coupling,
                            config.layoutgen_train_params.obj_bbx_conditioning,
                            config.layoutgen_train_params.use_fft_on_bbx,
                            config.layoutgen_train_params.use_gcn_in_decoder
                            )
    
    # Load model from a W&B artifact
    # Example to download weights from a specific run, if needed
    # artifact = wandb.use_artifact(f'project_name/model:latest')
    # artifact_path = artifact.download()
    # vae.load_state_dict(torch.load(os.path.join(artifact_path, 'model_weights.pth')))
    vae.load_state_dict(torch.load(config.layoutgen_test_params.weights_path))


    # decoder = vae.decoder
    image_shape = [config.layoutgen_model_params.num_nodes, config.layoutgen_model_params.bbx_size]
    global_step = 250000
    class_dict = {0:'cow', 1:'sheep', 2:'bird', 3:'person', 4:'cat', 5:'dog', 6:'horse', 7:'aeroplane',\
                8:'motorbike', 9:'bicycle', 10:'pottedplant'}
    res_dfs = []
    
    # Log test layouts
    # test_layouts_artifact = wandb.Artifact("test_layouts", type="test_layouts")
    
    for i, val_data in enumerate(batch_test_loader, 0):
        node_data_true = val_data.x
        label_true = node_data_true[:,:1]
        y_true = val_data.y
        class_true = y_true[:, :config.layoutgen_model_params.num_classes]
        X_obj_true = y_true[:, config.layoutgen_model_params.num_classes:]
        X_obj_true_transformed = X_obj_true[:,2:]-X_obj_true[:,:2]
        # X_obj_true_transformed = X_obj_true
        node_data_transformed = torch.cat(
                [node_data_true[:, :-2],
                node_data_true[:, -2:] -  node_data_true[:, -4:-2]], axis=-1)
        adj_true = val_data.edge_index
        class_true  = torch.flatten(class_true)
        
        output = vae(adj_true, node_data_true, X_obj_true_transformed, label_true , class_true, config.layoutgen_train_params.variational, \
                     config.layoutgen_train_params.coupling, refine_iter=config.layoutgen_train_params.refine_iter, training=False)
        
        node_data_pred_test = output[0]
        node_data_pred_test_new = torch.cat(
                [node_data_pred_test[:, :, :-2],
                node_data_pred_test[:, :, -2:] + node_data_pred_test[:, :, -4:-2]], axis=-1)
        X_obj_pred_test = output[1]
        X_obj_pred_test = torch.cat(((torch.tensor([1.0])-X_obj_pred_test)*torch.tensor([0.5]),
                                    (torch.tensor([1.0])+X_obj_pred_test)*torch.tensor([0.5])),-1)

        if config.layoutgen_train_params.use_gcn_in_decoder:
            node_data_pred_test_refined = output[8]
            node_data_pred_test_refined_new = torch.cat(
                    [node_data_pred_test_refined[:, :, :-2],
                    node_data_pred_test_refined[:, :, -2:] + node_data_pred_test_refined[:, :, -4:-2]], axis=-1)
            final_preds = node_data_pred_test_refined_new
        else:
            final_preds = node_data_pred_test_new 
        
        if config.layoutgen_test_params.plot_layouts:
            # Log layouts as images to a W&B artifact
            plot_utils.plot_pred_gt_layouts(config, node_data_true.cpu().detach().numpy(), final_preds.cpu().detach().numpy() , label_true.cpu().detach().numpy(), X_obj_pred_test.cpu().detach(), X_obj_true.cpu().detach(), y_true[:, :config.layoutgen_model_params.num_classes].cpu().detach().numpy())
            # cv2.imwrite(f'test_layout_{i}.png', gt_img)
            # test_layouts_artifact.add_file(f'test_layout_{i}.png')
            # REMOVED
            # plot_utils.plot_pred_gt_layouts(config, node_data_true.cpu().detach().numpy(), final_preds.cpu().detach().numpy() , label_true.cpu().detach().numpy(), X_obj_pred_test.cpu().detach(), X_obj_true.cpu().detach(), y_true[:, :config.layoutgen_model_params.num_classes].cpu().detach().numpy())
            # plot_utils.plot_layouts(config, final_preds.cpu().detach().numpy() , label_true.cpu().detach().numpy(), X_obj_pred_test.cpu().detach(), y_true[:, :config.layoutgen_model_params.num_classes].cpu().detach().numpy())
            

        if config.layoutgen_test_params.compute_metrics == True:
            res_dfs.append(metrics.get_metrics(node_data_true, X_obj_true, final_preds,
                                    X_obj_pred_test,
                                    label_true,
                                    class_true,
                                    config.layoutgen_model_params.num_nodes,
                                    config.layoutgen_model_params.num_classes))
            
            # REMOVED
            # if write_tensorboard:
            #     for j in range(int(len(node_data_true)/config.layoutgen_model_params.num_nodes)):
            #         image = plot_utils.plot_bbx(node_data_true[j].detach().to("cpu").numpy().astype(float))/255
            #         pred_image = plot_utils.plot_bbx(node_data_pred_test[j].detach().to("cpu").numpy()).astype(float)/255
            #         writer.add_image('test_result/images/'+str(j)+'-input/', image, global_step, dataformats='HWC')  
            #         writer.add_image('test_result/images/'+str(j)+'-generated/', pred_image, global_step, dataformats='HWC') 
            
                    

    if config.layoutgen_test_params.compute_metrics == True:
        result = pd.concat(res_dfs)
        result['obj_class'] = np.where(result['obj_class'].isna(), 0, result['obj_class'])
        result['obj_class'] = result['obj_class'].astype('int')
        result['obj_class'].replace(class_dict, inplace=True)
        result.where(result['part_labels']!=0, np.nan, inplace=True)
        result['part_labels'] = np.where(result['part_labels'].isna(), 0, result['part_labels'])
        result['part_labels'] = result['part_labels'].astype('int')
        result['id'] = np.repeat(np.array(list(range(int(len(result)/config.layoutgen_model_params.num_nodes)))), config.layoutgen_model_params.num_nodes)

        # Log metrics DataFrames as tables to W&B
        wandb.log({"raw_metrics_table": wandb.Table(dataframe=result)})
        res_obj_level = result.groupby(['obj_class', 'id']).mean(numeric_only=True).reset_index()
        obj_metrics = res_obj_level.groupby(['obj_class']).mean(numeric_only=True).reset_index()[['obj_class', 'IOU', 'MSE']]
        wandb.log({"obj_level_metrics_table": wandb.Table(dataframe=obj_metrics)})
        
        bird_labels = {'head':1 , 'torso':2, 'neck':3, 'lwing':4, 'rwing':5, 'lleg':6, 'lfoot':7, 'rleg':8, 'rfoot':9, 'tail':10}
        cat_labels = {'head':1, 'torso':2, 'neck':3, 'lfleg':4, 'lfpa':5, 'rfleg':6, 'rfpa':7, 'lbleg':8, 'lbpa':9, 'rbleg':10, 'rbpa':11, 'tail':12}
        cow_labels = {'head':1,'lhorn':2, 'rhorn':3, 'torso':4, 'neck':5, 'lfuleg':6, 'lflleg':7, 'rfuleg':8, 'rflleg':9, 'lbuleg':10, 'lblleg':11, 'rbuleg':12, 'rblleg':13, 'tail':14}
        dog_labels = {'head':1,'torso':2, 'neck':3, 'lfleg':4, 'lfpa':5, 'rfleg':6, 'rfpa':7, 'lbleg':8, 'lbpa':9, 'rbleg':10, 'rbpa':11, 'tail':12, 'muzzle':13}
        horse_labels = {'head':1,'lfho':2, 'rfho':3, 'torso':4, 'neck':5, 'lfuleg':6, 'lflleg':7, 'rfuleg':8, 'rflleg':9, 'lbuleg':10, 'lblleg':11, 'rbuleg':12, 'rblleg':13, 'tail':14, 'lbho':15, 'rbho':16}
        person_labels = {'head':1, 'torso':2, 'neck': 3, 'llarm': 4, 'luarm': 5, 'lhand': 6, 'rlarm':7, 'ruarm':8, 'rhand': 9, 'llleg': 10, 'luleg':11, 'lfoot':12, 'rlleg':13, 'ruleg':14, 'rfoot':15}
        sheep_labels = cow_labels
        part_labels_combined_parts = {'bird': bird_labels, 'cat': cat_labels, 'cow': cow_labels, 'dog': dog_labels, 'sheep': sheep_labels, 'horse':horse_labels,'person':person_labels}

        for k, v in part_labels_combined_parts.items():
            new_map = {}
            for pk, pv in v.items():
                new_map[pv]=pk
            part_labels_combined_parts[k] = new_map
            
        for k, v in part_labels_combined_parts.items():
            result.loc[result['obj_class']==k, ['part_labels']] = result.loc[result['obj_class']==k,['part_labels']].replace(v).copy()
        
        part_metrics = result.groupby(['obj_class', 'part_labels']).mean(numeric_only=True).reset_index()[['obj_class', 'part_labels',  'IOU', 'MSE']]
        # wandb.log({"part_level_metrics_table": wandb.Table(dataframe=part_metrics)})
        
        # REMOVED
        # if write_tensorboard:
        #     writer.flush()
        #     writer.close()
        result.to_csv(os.path.join(config.layoutgen_test_params.output_folder_path+'/raw_metrics.csv'))
        res_obj_level = result.groupby(['obj_class', 'id']).mean(numeric_only=True).reset_index()
        res_obj_level.groupby(['obj_class']).mean(numeric_only=True).reset_index()[['obj_class', 'IOU', 'MSE']].to_csv(os.path.join(config.layoutgen_test_params.output_folder_path, 'obj_level_metrics.csv'))
        # result.groupby(['obj_class', 'part_labels']).mean(numeric_only=True).reset_index()[['obj_class', 'part_labels',  'IOU', 'MSE']].to_csv(os.path.join(config.layoutgen_test_params.output_folder_path, 'part_level_metrics.csv'))
        print(f"Computed metrics...")

def make_schedule():
    nb_epochs = config.layoutgen_train_params.nb_epochs
    # one cycle: 0 → 1 inclusive in steps of 0.0004
    cycle = np.arange(0, 1 + 0.0004, 0.0004)
    # repeat cycle until reaching nb_epochs
    repeats = int(np.ceil(nb_epochs / len(cycle)))
    seq = np.tile(cycle, repeats)[:nb_epochs]

    return seq


if __name__ == '__main__':
    #NOTE: Disabled saving model weights for now. Enable this!! 
    #Parameters
    # kl_annealing_epochs = 1000
    k = 0.01
    t0 = 2000 
    config = OmegaConf.load("configs/layoutgen_config.yaml")

    klw = loss.frange_cycle_linear(config.layoutgen_train_params.nb_epochs)
    # klw = make_schedule()

    # Dataloaders 
    batch_train_loader, batch_val_loader = data_loading.load_data(config.layoutgen_train_params.data_path, obj_data_postfix = '_obj_boundary_sqr',
                                            part_data_post_fix = '_scaled_sqr',
                                            file_postfix = '_combined',
                                            seed=345,
                                            batch_size=config.layoutgen_model_params.batch_size)

    batch_test_loader = data_loading.load_data(config.layoutgen_test_params.data_path, obj_data_postfix = '_obj_boundary_sqr', \
                                                              part_data_post_fix = '_scaled_sqr',\
                                                              file_postfix = '_combined',\
                                                              seed=345,\
                                                              batch_size=config.layoutgen_model_params.batch_size, \
                                                              validation=False)
    print(f"len(batch_train_loader): {len(batch_train_loader)}")
    print(f"len(batch_val_loader): {len(batch_val_loader)}")
    print(f"len(batch_test_loader): {len(batch_test_loader)}")
    
    # Initialize W&B run here
    # wandb.init(project="palgo_training", name="pose_control_ckpt_disconn_loss_50", config=OmegaConf.to_container(config))

    wandb.init(mode="disabled")
    
    #Training
    if args.train:
        train()
    elif args.test:
        test()
    
    # End the W&B run
    wandb.finish()