import gradio as gr
import cv2 
from PIL import Image, ImageDraw, ImageFont
import torch
import yaml 
from openai import OpenAI
import ast 
import os 


import sys 
sys.path.append(f"/home/varghese/plato-gen/")
from e2e_pipeline import layout_gen
from e2e_pipeline import image_gen
from e2e_pipeline.utils import constants
from e2e_pipeline.utils import layout_plot_utils

import pdb 

_DEFAULT_CLASS = "bird"
_DEFAULT_PARTS = constants.ALL_PART_MAPPING[_DEFAULT_CLASS]


bbox_coords = None 
parts_ls = []


def get_object_parts(obj_class):
    return gr.CheckboxGroup(choices=list(list(constants.ALL_PART_MAPPING[obj_class].keys())), value=[])


def generate_layout_wrapper(obj_class, selected_parts, selected_pose):
    global parts_ls  
    if len(selected_pose) != 0:
        parts_ls = constants.ALL_PART_POSE[obj_class][selected_pose]
    else:
        parts_ls = selected_parts
    # obj_bbx, layout_bbx = layout_gen.generate_layout(_VAE_MODEL, obj_class, parts_ls)
    obj_bbx, layout_bbx = layout_gen.generate_layout_gcn(config, _VAE_MODEL, obj_class, parts_ls)
    layout_ellipse, layout_rect, _ = layout_plot_utils.plot_ellipse(obj_bbx, layout_bbx)
    return layout_ellipse, layout_rect


def create_legend(layout_bbx, obj_class, selected_parts):
    legend_labels = {i: constants.colors[constants.ALL_PART_MAPPING[obj_class][i]] for i in selected_parts}
    #Create legend image
    legend_width, legend_height = 500, 100
    legend_image = Image.new("RGB", (legend_width, legend_height), "white")
    draw = ImageDraw.Draw(legend_image)
    # Font settings (use default if custom font unavailable)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default(size=16)

    # Spacing and dimensions for legend items
    box_size = 30
    padding = 10
    x_offset, y_offset = padding, padding
    row_height = box_size + padding

    for label, color in legend_labels.items():
        # Measure the size of the text
        # text_width, text_height = draw.textlength(label, font=font)
        text_width = draw.textlength(label, font=font)
        item_width = box_size + padding + text_width + padding

        # Check if the item fits in the current row
        if x_offset + item_width > legend_width:
            # Move to the next row
            x_offset = padding
            y_offset += row_height

        # Draw the color box
        draw.rectangle([x_offset, y_offset, x_offset + box_size, y_offset + box_size], fill=tuple(color.astype(int)))

        # Draw the label text
        draw.text((x_offset + box_size + padding, y_offset), label, fill="black", font=font)

        # Update x_offset for the next item
        x_offset += item_width

    return legend_image


def generate_image(obj_class, selected_parts):
    #Layout
    layout_image, layout_bbx = generate_layout_wrapper(obj_class, selected_parts, [])
    #Legend
    legend_image = create_legend(layout_bbx, obj_class, selected_parts)
    object_image, _, _, _ = image_gen.generate_image(_SD_PIPELINE, obj_class, layout_image, parts_ls)
    object_image = object_image.resize((1000, 1000), Image.Resampling.LANCZOS)
    return layout_bbx, layout_image, object_image, legend_image, gr.CheckboxGroup(choices=list(list(constants.ALL_PART_MAPPING[obj_class].keys())), value=[])



def clear_images(obj_class):
    return None, None, None, gr.CheckboxGroup(choices=list(list(constants.ALL_PART_MAPPING[obj_class].keys())), value=[]), \
                                    gr.Radio(choices=list(constants.ALL_PART_MAPPING.keys()), value=_DEFAULT_CLASS, label="Select Object")
            


def main():
    with gr.Blocks() as demo:

        with gr.Row():
            with gr.Column(scale=1, min_width=400):
                object_class = gr.Radio(choices=list(constants.ALL_PART_MAPPING.keys()), value=_DEFAULT_CLASS, label="Select Object")
                # selected_pose = gr.Radio(choices=["Left_pose", "Right_pose"], value=[], label="Select Pose")
                selected_parts = gr.CheckboxGroup(choices=list(_DEFAULT_PARTS.keys()), value=[], label="Select Parts")
                button_image_gen = gr.Button('Generate Image!')
                button_image_clear = gr.Button('Clear Image')
                ##
                with gr.Accordion("Show Generated Layout and Control Image", open=False):
                    with gr.Row():
                        layout_bbx = gr.Image(f"./src/samples/9.png", label="Layout", width=300, height=300)
                        layout_image = gr.Image(f"./src/samples/9.png", label="Control Image", width=300, height=300)
                
                with gr.Row():
                    legend_image = gr.Image(f"./src/samples/9.png", label="legend", width=500, height=100)


            
            with gr.Column(scale=1, min_width=800):
                object_image = gr.Image(label="Object", width=800, height=800)




        object_class.change(get_object_parts, inputs=[object_class], outputs=[selected_parts])
        button_image_gen.click(generate_image, inputs=[object_class, selected_parts], outputs=[layout_bbx, layout_image, object_image, legend_image, selected_parts])
        button_image_clear.click(clear_images, inputs=[object_class], outputs=[layout_bbx, layout_image, object_image, selected_parts, object_class])

    demo.launch(share=True)

if __name__ == '__main__':
    #Create temp folder for gradio
    os.environ["GRADIO_TEMP_DIR"] = "./gradio_tmp"
    os.makedirs(f"./gradio_tmp", exist_ok=True)
    #Load Configs
    with open(f"./configs/layoutgen_config.yaml", "r") as f:
        layoutgen_cfg = yaml.safe_load(f)
    with open(f"./configs/imagegen_config.yaml", "r") as f:
        imagegen_cfg = yaml.safe_load(f)
    with open(f"./configs/e2e_config.yaml", "r") as f:
        e2e_cfg = yaml.safe_load(f)
    
    config = layoutgen_cfg | imagegen_cfg | e2e_cfg
    #Load Models
    _VAE_MODEL = layout_gen.load_vae(config)
    _SD_PIPELINE = image_gen.load_sd_model(config)

    main()
