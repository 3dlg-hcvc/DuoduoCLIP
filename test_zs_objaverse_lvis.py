import csv
import json
import h5py
import torch
import random
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

architecture = 'ViT-B-32'
pretrained_checkpoint = 'laion2b_s34b_b79k'

mv_data_h5 = 'dataset/data/lvis_images.h5'
data_list_path = 'dataset/csv_list/lvis.csv'
model_to_idx_path = 'dataset/data/lvis_model_to_idx.json'
clip_feat_path = 'dataset/data/lvis_class_label_embeddings.npy'

class LVIS(Dataset):
    def __init__(self, num_views, mv_data_h5, data_list_path, clip_feat_path, model_to_idx_path, preprocess):
        super().__init__()
        self.num_views = num_views
        self.preprocess = preprocess
        self.data_list_path = data_list_path
        self.clip_feat_path = clip_feat_path
        self.model_to_idx_path = model_to_idx_path
        
        self.data_info = self._read_data_info()
        self.mv_images_h5 = h5py.File(mv_data_h5, "r")
        self.clip_cat_feat = np.load(clip_feat_path, allow_pickle=True)

        with open(self.model_to_idx_path, 'r') as file:
            self.model_to_idx = json.load(file)

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        model_id = self.data_info[idx]["model_id"]
        mv_images = self._process_mv_data(model_id)

        return {
            "mv_images": mv_images,
            "class_idx": self.data_info[idx]["class_idx"],
            "category_clip_features": self.clip_cat_feat
        }

    def _read_data_info(self):
        data_info = []
        with open(self.data_list_path, "r") as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data_info.append(
                    {
                        "class_idx": int(line[0]),
                        "model_id": line[-1],
                    }
                )
        return data_info

    def _process_mv_data(self, model_id):
        image_idx = self.model_to_idx[model_id]
        mv_images = self.mv_images_h5["imgs"][image_idx]

        selected_numbers = random.sample(range(12), self.num_views)
        mv_images = mv_images[selected_numbers]

        mv_images_processed = torch.empty(size=mv_images.shape, dtype=torch.float16)
        for i, single_view_img in enumerate(mv_images):
            mv_images_processed[i] = self.preprocess(Image.fromarray(single_view_img.transpose(1, 2, 0), mode="RGB")).to(torch.float16)

        return mv_images_processed

def _collate_fn(batch):
    default_collate_items = ("mv_images", "class_idx")
    batch_data = []
    for i, b in enumerate(batch):
        batch_data.append({k: b[k] for k in default_collate_items})
    data_dict = default_collate(batch_data)
    data_dict["category_clip_features"] = torch.from_numpy(batch[0]["category_clip_features"])
    return data_dict

@torch.no_grad()
@torch.autocast(device_type="cuda")
def generate_clip_image_embeddings(clip_model, batch_imgs):
    return clip_model.encode_image(batch_imgs)

open_clip_model, _, preprocess = open_clip.create_model_and_transforms(architecture, pretrained=pretrained_checkpoint, device='cuda', precision="fp16")

all_results = {}
for num_views in [1, 2, 3, 4, 5, 6, 8, 10, 12]:
    objaverse_lvis_dataset = LVIS(num_views=num_views, mv_data_h5=mv_data_h5, data_list_path=data_list_path, clip_feat_path=clip_feat_path, model_to_idx_path=model_to_idx_path, preprocess=preprocess)
    objaverse_lvis_dataloader = DataLoader(objaverse_lvis_dataset, batch_size=200, pin_memory=True, num_workers=8, drop_last=False, collate_fn=_collate_fn)

    top_ks = (1, 3, 5)
    tok_k_acc = {}
    for k in top_ks:
        tok_k_acc[k] = Accuracy(task="multiclass", num_classes=1156, top_k=k).cuda()

    for i, item in enumerate(tqdm(objaverse_lvis_dataloader)):
        img_embeddings = generate_clip_image_embeddings(open_clip_model, item['mv_images'].flatten(0, 1).half().cuda())
        img_embeddings = img_embeddings.reshape(-1, num_views, open_clip_model.visual.output_dim).mean(1)
        logits = F.normalize(img_embeddings, dim=1) @ F.normalize(item['category_clip_features'].half().cuda(), dim=1).T

        for k in top_ks:
            tok_k_acc[k].update(logits, item["class_idx"].cuda())

    # print the results
    line_width = 60
    print("Objaverse LVIS Test Results:")
    print('=' * line_width)
    print(" | ".join([f"{header:<5}" for header in [f"Top-{top_k}" for top_k in top_ks]]) + " |")
    print('-' * line_width)

    formatted_accuracies = ["{:<5}".format(round(tok_k_acc[k].compute().cpu().item() * 100, 2)) for k in top_ks]
    print(" | ".join(formatted_accuracies) + " |")

    results = [round(tok_k_acc[k].compute().cpu().item() * 100, 2) for k in top_ks]
    all_results[num_views] = results
