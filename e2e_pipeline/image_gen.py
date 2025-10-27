from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler, UNet2DConditionModel
import numpy as np
import PIL
import torch
import cv2 
import pdb 

from e2e_pipeline.utils import data_preprocessor
from e2e_pipeline.utils import constants 

import pickle

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_sd_model(config):
  controlnet = ControlNetModel.from_pretrained(config["e2e_pipeline"]["imagegen_weights"], torch_dtype=torch.float32)
  pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", \
                                                                    controlnet=controlnet,safety_checker=None)

  pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
  pipe = pipe.to(device)
  pipe.enable_model_cpu_offload()
  return pipe


def process_cond_image(cond_image, mask_image_coords):
    cond_image = np.array(cond_image)
    mask_height = mask_image_coords[3] - mask_image_coords[1]
    mask_width = mask_image_coords[2] - mask_image_coords[0]
    non_black_pixels = np.where(np.any(cond_image != [0, 0, 0], axis=-1))
    # Calculate the bounding box
    xmin = np.min(non_black_pixels[1])  # Minimum x-coordinate
    ymin = np.min(non_black_pixels[0])  # Minimum y-coordinate
    xmax = np.max(non_black_pixels[1])  # Maximum x-coordinate
    ymax = np.max(non_black_pixels[0])  # Maximum y-coordinate

    cond_image_cropped = cond_image[ymin:ymax, xmin:xmax]
    cond_image_cropped = cv2.resize(cond_image_cropped, (mask_width, mask_height))

    black_canvas = np.zeros((512, 512, 3), dtype=np.uint8)
    black_canvas[mask_image_coords[1]: mask_image_coords[1] + mask_height, \
                mask_image_coords[0]: mask_image_coords[0] + mask_width] = cond_image_cropped
    return black_canvas


def generate_image(pipe, obj_class, control_image, parts_list, subspecies_class=None, save_iter=0):
  
  if subspecies_class is None:
    subspecies_class = obj_class 

  init_image = np.zeros((512,512,3))*255
  # init_image = PIL.Image.open(f"/home/varghese/palgo/application/init_image.jpg")
  control_image = PIL.Image.fromarray(control_image)
  # control_image = PIL.Image.open(f"/home/varghese/palgo/application/control_image.jpg")
  # control_image = process_cond_image(control_image, [156, 113, 362, 364])
  # control_image = PIL.Image.fromarray(control_image)
  # mask_image = PIL.Image.open(f"/home/varghese/palgo/application/mask_image.jpg")
  mask_image = data_preprocessor.binarize_to_bbx(control_image.copy())
  # mask_image = PIL.Image.open(f"/home/varghese/palgo/application/mask_image.jpg")
  try:
    parts_list_expanded = [constants.ALL_EXPANDED_LABELS_MAPPING[obj_class][i] for i in parts_list]
  except:
    parts_list_expanded = parts_list 
  
  #Caption for image generation
  # subspecies_class = f"black horse"
  prompt = f"A segmented sprite of a real {subspecies_class} with photo-realistic details and body structure on a black background."
  # prompt = f"A segmented sprite of a real parrot with photo-realistic details and body structure on a white background."
  prompt += " Image shows: "
  for part in parts_list_expanded:
    if obj_class == "pottedplant" and part == "body":
        continue
    if obj_class == "aeroplane" and (part == "engine" and "engine" in prompt) or (part == "wheel" and "wheel" in prompt):
        continue
    if obj_class == "motorbike" and part == "headlight" and "headlight" in prompt:
        continue
    prompt += f"{subspecies_class}'s {part}, "
  prompt = prompt[:-2]+"."
  print(f"{prompt = }")
  
  output = pipe(
    prompt,
    negative_prompt= f"low quality, 3d, pixelated, cartoonish",
    image=init_image,
    control_image=control_image,
    mask_image=mask_image).images[0]
  # save_iter += 1
  # return output, control_image, prompt, init_latent.detach().cpu()
  return output, None, None, None 