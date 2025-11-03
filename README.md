<div align="center">

<samp>
<h1>PLATO</h1>
<h2>Generating Objects from Part Lists via Synthesized
Layouts </h2>
</samp>


**_[ACMM 2025]_**

| [![PDF + arXiv](https://img.shields.io/badge/PDF%20|%20arXiv-brightgreen)](https://link_to_paper.com) | [![Project + Homepage](https://img.shields.io/badge/Project%20|%20Homepage-ff5f5f)](https://plato-gen.github.io/) | [![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/plato-gen/plato-gen/blob/main/LICENSE) |



</div>

---

# 📚 Table of Contents

1. [Overview](#plato-overview)
2. [Getting Started](#-getting-started)
3. [Demo](#-demo)
4. [Image generation: Training and Inference](#-image-generation:-training-and-inference)
5. [Layout generation: Training and Inference](#-layout-generation-training-and-inference)
6. [Citation](#-citation)
7. [Contact](#-contact)

---

# PLATO Overview

We introduce PLATO, a novel two-stage framework that enables precise, part-controlled object generation. The first stage is PLayGen, our novel part layout generator which takes a list of parts and object category as input and synthesizes high fidelity layouts of part bounding boxes.PLayGen’s synthesized layout is
used to condition a custom-tuned ControlNet-style adapter, enforcing spatial and connectivity constraints.

---

# 🛠 Getting Started

### Installation
We use a conda environment with the following:
- Python 3.9
- PyTorch 2.5.1
- CUDA 12.4

Rest of the dependencies can be installed using the following command
```bash
$ pip install -r requirements.txt
```

### ✅ Model Checkpoints

We provide three ready-to-use model checkpoints fine-tuned on the Pascal-Parts dataset that can be used out of the box.

| Model            | Download Link |
|------------------|----------------------|
| **PLayGen**     | [Download](https://drive.google.com/file/d/1I_30xgVhyKDuZlxZAJqBNXDPfbU4lDdZ/view?usp=sharing) |
| **Image Generator** | [Download](https://huggingface.co/VargheseP/Mero_sd2_full_ellipse)|

---
# 🚀 Demo
We include a gradio app which can be run with the following command:
```
$ python app.py
```
---

# 🖼️ Image Generation: Training and Inference

### ImageGen Inference
Configuration for inference is present in configs/imagegen_config.yaml. 
```
$ python imagegen/test_imagegen.py
```

### ImageGen Training
- We use the ControlNet training script from the diffusers library. For more details refer to [Train your controlnet](https://huggingface.co/blog/train-your-controlnet). 
```bash 
$ export MODEL_DIR="stabilityai/stable-diffusion-2-1"
$ export CONTROLNET_DIR="thibaud/controlnet-sd21-ade20k-diffusers"
$ export OUTPUT_DIR="" #Enter path directory where ckpts are saved
$ accelerate launch train_imagegen.py --pretrained_model_name_or_path=$MODEL_DIR --controlnet_model_name_or_path=$CONTROLNET_DIR --output_dir=$OUTPUT_DIR --dataset_name=VargheseP/palgo_ellipse_new_train --lr_scheduler="constant_with_warmup" --lr_warmup_step=100 --train_batch_size=16 --gradient_accumulation_steps=2 --gradient_checkpointing --enable_xformers_memory_efficient_attention --set_grads_to_none --caption_column="caption_wt_parts" --mask_column="mask_image" --mixed_precision fp16 --use_8bit_adam --num_train_epochs=100
```

# Layout Generation: Training and Inference
The config/layoutgen_config.yaml file contains the configuration for training and testing.
- For Training
```
python playgen/train.py --train
```
- For Testing
```
python playgen/train.py --test
```
---

# 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{yourcitation2025, 
  title={PLATO: Generating Objects from Part Lists via Synthesized Layouts}, 
  author={Amruta Muthal, Varghese P Kuruvilla,  Ravi Kiran Sarvadevabhatla}, 
  booktitle={ACMM}, 
  year={2025}
}
```

---

## 📬 Contact

For issues in running code/links not working, please reach out to [Amruta Muthal](mailto:amruta.muthal@research.iiit.ac.in) or [Varghese P Kuruvilla](mailto:varghese.kuruvilla@research.iiit.ac.in) or mention in the **ISSUES** section.

For questions or collaborations, please reach out to [Dr. Ravi Kiran Sarvadevabhatla](mailto:ravi.kiran@iiit.ac.in).
