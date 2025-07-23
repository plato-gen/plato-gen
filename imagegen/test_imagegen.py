'''
Script to perform Image generation inference on a test dataset
'''
from diffusers import StableDiffusionControlNetPipeline, StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import datasets 
import numpy as np 
import cv2, os, PIL  
import glob 
from PIL import Image 
from omegaconf import OmegaConf
from torchvision import transforms

##For debugging
import pdb

def create_overlay_images(image1, image2):
    image1 = image1.convert("RGBA")
    image2 = image2.convert("RGBA")
    image2.putalpha(128)
    combined_img = Image.alpha_composite(image1, image2)
    return combined_img

class PalgoInf:
    def __init__(self,config):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = config 
        controlnet = ControlNetModel.from_pretrained(config.imagegen_inference_params.controlnet_model_path, torch_dtype=torch.float32)
        self.pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(config.imagegen_inference_params.base_model_path, controlnet=controlnet, torch_dtype=torch.float32, safety_checker=None)
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe = self.pipe.to(self.device)
        print(f"Completed init")

    def inference_single(self, init_image, cond_image, mask_image, prompt, negative_prompt):
        gen_image = self.pipe(prompt, negative_prompt=negative_prompt, image=init_image, control_image=cond_image, mask_image=mask_image).images[0]
        return gen_image 

    def inference_hf_dataset(self, test_dataset, cond_type, prompt_type, negative_prompt):
        """
        Inference on a hf dataset. The dataset has columns: image, conditioning_image, mask_image, caption_wt_parts
        """
        init_image = np.zeros((512, 512, 3)) * 255
        init_image = init_image.astype(np.uint8)
        init_image = PIL.Image.fromarray(init_image)
        split = 'test'
        for row_iter in range(0, len(test_dataset[split])):  
            gt_image, conditioning_image, mask_image, caption  = test_dataset[split][row_iter]['image'], test_dataset[split][row_iter]['conditioning_image'], test_dataset[split][row_iter]['mask_image'], \
                                                                 test_dataset[split][row_iter]['caption_wt_parts']
            if row_iter % 10 == 0:
                print(f"Processing {row_iter}")
                print(f"{caption = }")

            if conditioning_image.size != (512, 512):
                conditioning_image = resize_transform(conditioning_image)
            if gt_image.size != (512, 512):
                gt_image = resize_transform(gt_image)
            gen_image = self.inference_single(init_image, conditioning_image, mask_image, caption, negative_prompt)
            gen_image.save(os.path.join(self.config.imagegen_inference_params.output_image_folder, f"gen_{row_iter}.png"))
            images_overlay = create_overlay_images(gen_image, conditioning_image)
            images_overlay.save(os.path.join(self.config.imagegen_inference_params.output_image_folder, f"overlay_{row_iter}.png"))
            np_gen_image, np_conditioning_image, np_mask_image = np.array(gen_image), np.array(conditioning_image), np.array(mask_image)

            if row_iter == 0:
                images_concat = np.expand_dims(np_gen_image, axis=0)
            else:
                images_concat = np.concatenate((images_concat, np.expand_dims(np_gen_image, axis=0)), axis=0)
        
        #Save as numpy file for further processing
        np.save(self.config.imagegen_inference_params.output_npfile_path, images_concat)
        print(f"Completed inference")

   





if __name__ == '__main__':
    #Load config
    config = OmegaConf.load(f"configs/imagegen_config.yaml")
    #Define transforms
    resize_transform = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR)
    #Test dataset
    test_dataset = datasets.load_dataset(config.imagegen_inference_params.test_dataset_name)
    print(f"Loaded test_dataset")
    PalgoInf_obj = PalgoInf(config)
    PalgoInf_obj.inference_hf_dataset(test_dataset, cond_type="conditioning_image", prompt_type="caption_wt_parts",\
                                  negative_prompt=f"low quality, 3d, pixelated, cartoonish")
    
    
