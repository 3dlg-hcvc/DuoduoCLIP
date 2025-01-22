# Duoduo CLIP: Efficient 3D Understanding with Multi-View Images

[Han-Hung Lee<sup>*1</sup>](https://hanhung.github.io/), 
[Yiming Zhang<sup>*1</sup>](https://scholar.google.com/citations?user=scUaE38AAAAJ&hl=en) and 
[Angel Xuan Chang<sup>1,2</sup>](https://angelxuanchang.github.io/)

<sup>*</sup> Equal Contribution <sup>1</sup> Simon Fraser University <sup>2</sup> Canada-CIFAR AI Chair, Amii

<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/Lightning-792DE4?style=for-the-badge&logo=pytorch-lightning&logoColor=white"></a>
<a href="https://wandb.ai/site"><img alt="WandB" src="https://img.shields.io/badge/Weights_&_Biases-FFBE00?style=for-the-badge&logo=WeightsAndBiases&logoColor=white"></a>

**[Project Page](https://3dlg-hcvc.github.io/DuoduoCLIP/)**

## Abstract

> We introduce Duoduo CLIP, a model for 3D representation learning that learns shape encodings from multi-view images instead of point-clouds. 
> The choice of multi-view images allows us to leverage 2D priors from off-the-shelf CLIP models to facilitate fine-tuning with 3D data. 
> Our approach not only shows better generalization compared to existing point cloud methods, but also reduces GPU requirements and training time. 
> In addition, we modify the model with cross-view attention to leverage information across multiple frames of the object which further boosts performance. 
> Compared to the current SOTA point cloud method that requires 480 A100 hours to train 1 billion model parameters we only require 57 A5000 hours and 87 million parameters.
> Multi-view images also provide more flexibility in use cases compared to point clouds.
> This includes being able to encode objects with a variable number of images, with better performance when more views are used.
> This is in contrast to point cloud based methods, where an entire scan or model of an object is required.
> We showcase this flexibility with object retrieval from images of real-world objects. Our model also achieves better performance in more fine-grained text to shape retrieval, demonstrating better text-and-shape alignment than point cloud based models.

## Notes

This is the official initial release for the paper **Duoduo CLIP: Efficient 3D Understanding with Multi-View Images**. In this release we provide evaluation for the LVIS split of Objaverse as well object retrieval from text. We will release the entire data preparation and training code soon. See [TODOs](#todos) for items we will add to the repo. The pretrained models as well as model cards will be provided in this [repo](https://huggingface.co/3dlg-hcvc/DuoduoCLIP) and the data [here](https://huggingface.co/datasets/3dlg-hcvc/DuoduoCLIP-data).

## Environment Setup

### Conda
> We use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage system dependencies.
```bash
# create and activate the conda environment
conda create -n ddclip python=3.10
conda activate ddclip

# install PyTorch
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# install Python libraries
pip install -r requirements.txt
cd open_clip_mod
pip install .

# install Faiss
conda install -c pytorch -c nvidia faiss-gpu=1.8.0
```

## Example

```python
import numpy as np
from PIL import Image

from src.model.wrapper import get_model

mv_images = Image.open('examples/couch.jpg')
mv_images = np.asarray(mv_images).reshape(12, 224, 224, 3)

duoduoclip = get_model('Four_1to6F_bs1600_LT6.ckpt', device='cuda')
text_features = duoduoclip.encode_text('a 3D model of a white couch')

# The model can take multi-view images of shape (F, H, W, 3)
# (F is number of multi-views, H and W are image resolutions)
image_features = duoduoclip.encode_image(mv_images)
similarity = text_features.squeeze() @ image_features.squeeze()
print(similarity)

# The model can also take single view images of shape (H, W, 3)
# (H and W are image resolutions)
image_features = duoduoclip.encode_image(mv_images[0])
similarity = text_features.squeeze() @ image_features.squeeze()
print(similarity)
```

## Dataset

### Objaverse LVIS

1. Download the objaverse lvis files (~80GB) for evaluation and placed in the ***dataset/data*** folder.
```bash
python preprocess/download_lvis.py
```

### Preprocessed Objaverse Embeddings
We also provide embeddings for each object of the entire objaverse dataset using 12 randomly rendered views for each object.

1. Download the shape embeddings (~800M). This includes the shapes embeddings produced by the default model placed under ***dataset/data/objaverse_embeddings/Four_1to6F_bs1600_LT6***.
```bash
python preprocess/download_embeddings.py
```

### Training Data

### Four Dataset
Note this will take a large amount of disk space to store as there is about 800k renderings of objects each with 12 views at resolution 224x224.

1. Download the multi-view objaverse renderings from **[Zero123](https://github.com/cvlab-columbia/zero123)** (~1.5TB) and uncompress.
```bash
wget https://tri-ml-public.s3.amazonaws.com/datasets/views_release.tar.gz
```
2. Download supplement multi-view images (~160GB) not included in Zero123 as well as image and text embeddings used in training.
```bash
python preprocess/download_supplement.py
```
3. Merge multi-view images into an h5 file. Resulting h5 file is ~1.5TB.
```bash
python preprocess/combine_four_h5.py --zero123_path /path/to/uncompressed/views_release
```

## Evaluation

### Objaverse

1. Run the objaverse lvis evaluation over multiple view settings. The model here is trained with 1 to 6 frames sampled during training with last 6 layers trainable.
```bash
python test_objaverse_lvis.py ckpt_path=Four_1to6F_bs1600_LT6.ckpt
```

## Retrieval

1. Retrieve objaverse models using text as input. **You can visualize models [here](https://objaverse.allenai.org/explore).**
```bash
python text_retrieval.py ckpt_path=Four_1to6F_bs1600_LT6.ckpt
```

## Training

### Four

- Train model with command. (Note this requires 4 gpus with at least 24GB of memory)
```bash
python train.py experiment_name=Four_1to6F_bs1600_LT6 trainer.devices=4
```

- Important Flags
```python
# Number of GPUs
trainer.devices=4

# Training batch size
data.train.dataloader.batch_size=400

# Number of multi-views to sample during training
"data.train.metadata.num_views=[1, 6]"

# Layer threshold to train (here all layers after and including the 6th layer will be trained, set to 0 to train all layers)
model.network.layers_threshold=6
```

## TODOs

- [ ] Add data preparation code for Four, MVImgNet and Text2Shape.
- [ ] Add training code for all setting in the paper.
- [ ] Add evaluation scripts for MVPNet and Text2Shape.

## Acknowledgements

### Code

[**OpenCLIP**](https://github.com/mlfoundations/open_clip): Our model backbones and weights are based off the open source implementation OpenCLIP. The folder ***open_clip_mod*** contains the same code as in the [OpenCLIP](https://github.com/mlfoundations/open_clip), but with some minor modifications to expose some additional functions from the package. The code within ***src/custom_clip*** modifies the OpenCLIP models to support the multi-view attention as described in the paper.

[**OpenShape**](https://github.com/Colin97/OpenShape_code): Our training framework closely follows that of OpenShape. We also use their provided model ids and text captions of their released [dataset](https://huggingface.co/datasets/OpenShape/openshape-training-data) for training.

[**Zero123**](https://github.com/cvlab-columbia/zero123): A large chunk of our rendered images for objects come from the paper Zero123, we also use their rendering script to render images for remaining objects.

We thank the authors for their work and releasing their code and weights!

## Citing
```
@misc{lee2024duoduo,
      title={Duoduo CLIP: Efficient 3D Understanding with Multi-View Images}, 
      author={Han-Hung Lee and Yiming Zhang and Angel X. Chang},
      year={2024},
      eprint={2406.11579},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Funding

This work was funded by a CIFAR AI Chair, a NSERC Discovery grant, and a CFI/BCKDF JELF grant.
