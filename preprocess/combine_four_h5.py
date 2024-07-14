import os
import json
import h5py
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class Zero123Dataset(Dataset):
    def __init__(self, zero123_path, zero123_model_ids):
        self.zero123_path = zero123_path
        self.zero123_model_ids = zero123_model_ids

    def __len__(self):
        return len(self.zero123_model_ids)

    def __getitem__(self, idx):
        model_id = self.zero123_model_ids[idx]
        object_path = os.path.join(args.zero123_path, model_id)

        images = []
        for i in range(12):
            filename = f"{i:03d}.png"
            img = Image.open(os.path.join(object_path, filename))
            img = img.resize((224, 224))
            img = np.asarray(img) / 255
            img = img[..., :3] * img[..., -1:] + (1 - img[..., -1:])
            images.append((img * 255).astype(np.uint8))
        images = np.stack(images, axis=0).transpose(0, 3, 1, 2)

        return {'images': images, 'model_id': model_id}

def _collate_fn(batch):
    batch_data = []
    for _, b in enumerate(batch):
        batch_data.append((b['images'], b['model_id']))
    return batch_data


parser = argparse.ArgumentParser()
parser.add_argument("--zero123_path", type=str, required=True, help="Path to zero123 render folder")
args = parser.parse_args()

with open('dataset/data/model_to_idx.json', 'r') as file:
    model_to_idx = json.load(file)
all_model_ids = list(model_to_idx.keys())

with open('dataset/data/supplement_model_to_idx.json', 'r') as file:
    supplement_model_to_idx = json.load(file)
supplement_model_ids = list(supplement_model_to_idx.keys())
zero123_model_ids = list(set(all_model_ids).difference(set(supplement_model_ids)))

assert len(all_model_ids) == len(supplement_model_ids) + len(zero123_model_ids)

with h5py.File("dataset/data/mv_images.h5", "w") as h5_file:
    h5_file.create_dataset("imgs", shape=(len(all_model_ids), 12, 3, 224, 224), dtype=np.uint8)

    supplement_h5 = h5py.File("dataset/data/supplement_mv_images.h5", "r")
    for model_id in tqdm(supplement_model_ids):
        images = supplement_h5['imgs'][supplement_model_to_idx[model_id]]
        h5_file['imgs'][model_to_idx[model_id]] = images
    supplement_h5.close()

    batch_size = 10
    dataset = Zero123Dataset(args.zero123_path, zero123_model_ids)
    zero123_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=8, drop_last=False, collate_fn=_collate_fn)

    for batch in tqdm(zero123_dataloader):
        for data in batch:
            images = data[0]
            model_id = data[1]
            h5_file['imgs'][model_to_idx[model_id]] = images
