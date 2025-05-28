import torch
import open_clip
import numpy as np
from tqdm import tqdm

# Get class id to category
class_dict = {}
with open('captions/mvimgnet_category.txt', 'r') as file:
    for line in file:
        line = line.strip()
        if line:
            idx, class_name = line.split(',')
            class_dict[int(idx)] = class_name

# Get new class ids so that there are no gaps
relabeled_class_dict = {}
for counter, category in enumerate(class_dict.values()):
    relabeled_class_dict[category] = counter
assert list(relabeled_class_dict.values()) == [i for i in range(len(relabeled_class_dict))]

# ULIP templates for getting embeddings for classes
ULIP_TEMPLATES = {
    "shapenet_64": [
        "a point cloud model of {}.",
        "There is a {} in the scene.",
        "There is the {} in the scene.",
        "a photo of a {} in the scene.",
        "a photo of the {} in the scene.",
        "a photo of one {} in the scene.",
        "itap of a {}.",
        "itap of my {}.",
        "itap of the {}.",
        "a photo of a {}.",
        "a photo of my {}.",
        "a photo of the {}.",
        "a photo of one {}.",
        "a photo of many {}.",
        "a good photo of a {}.",
        "a good photo of the {}.",
        "a bad photo of a {}.",
        "a bad photo of the {}.",
        "a photo of a nice {}.",
        "a photo of the nice {}.",
        "a photo of a cool {}.",
        "a photo of the cool {}.",
        "a photo of a weird {}.",
        "a photo of the weird {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of a clean {}.",
        "a photo of the clean {}.",
        "a photo of a dirty {}.",
        "a photo of the dirty {}.",
        "a bright photo of a {}.",
        "a bright photo of the {}.",
        "a dark photo of a {}.",
        "a dark photo of the {}.",
        "a photo of a hard to see {}.",
        "a photo of the hard to see {}.",
        "a low resolution photo of a {}.",
        "a low resolution photo of the {}.",
        "a cropped photo of a {}.",
        "a cropped photo of the {}.",
        "a close-up photo of a {}.",
        "a close-up photo of the {}.",
        "a jpeg corrupted photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of a {}.",
        "a blurry photo of the {}.",
        "a pixelated photo of a {}.",
        "a pixelated photo of the {}.",
        "a black and white photo of the {}.",
        "a black and white photo of a {}",
        "a plastic {}.",
        "the plastic {}.",
        "a toy {}.",
        "the toy {}.",
        "a plushie {}.",
        "the plushie {}.",
        "a cartoon {}.",
        "the cartoon {}.",
        "an embroidered {}.",
        "the embroidered {}.",
        "a painting of the {}.",
        "a painting of a {}."
    ]
}

@torch.no_grad()
@torch.autocast(device_type="cuda")
def generate_clip_text_embeddings(clip_model, batch_texts, tokenizer, device):
    batch_tokens = tokenizer(batch_texts).to(device)
    return clip_model.encode_text(batch_tokens).cpu().numpy()

reverse_relabeled_class_dict = {}
for key, value in relabeled_class_dict.items():
    reverse_relabeled_class_dict[value] = key
num_classes = len(reverse_relabeled_class_dict)

open_clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda', precision="fp16")
tokenizer = open_clip.get_tokenizer('ViT-B-32')

class_label_feat = np.empty(shape=(num_classes, open_clip_model.visual.output_dim), dtype=np.float16)
for i in tqdm(range(num_classes)):
    template_sentences = [template_sentence.format(reverse_relabeled_class_dict[i]) for template_sentence in ULIP_TEMPLATES["shapenet_64"]]
    text_embeddings = generate_clip_text_embeddings(clip_model=open_clip_model, batch_texts=template_sentences, tokenizer=tokenizer, device='cuda').mean(axis=0)
    class_label_feat[i] = text_embeddings

np.save('data/mvimgnet_class_label_embeddings.npy', class_label_feat)
