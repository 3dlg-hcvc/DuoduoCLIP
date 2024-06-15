import json
import h5py
import random
import numpy as np
from torch.utils.data import Dataset


class Text2Shape(Dataset):

    def __init__(self, data_cfg):
        super().__init__()
        self.data_cfg = data_cfg
        self.model_id_to_text_features = np.load(self.data_cfg.metadata.text2shape_clip_feat_path, allow_pickle=True).item()

        with open(self.data_cfg.metadata.text2shape_model_ids, 'r') as file:
            self.text2shape_model_ids = file.readlines()
        self.text2shape_model_ids = [line.strip() for line in self.text2shape_model_ids if line.strip()]
        self.model_ids = self.text2shape_model_ids

        with open(self.data_cfg.metadata.model_to_idx, 'r') as file:
            self.model_to_idx = json.load(file)
        self.mv_images_h5 = h5py.File(self.data_cfg.metadata.mv_data_h5, "r")

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        model_id = self.model_ids[idx]
        true_model_id = model_id.split('_')[0]

        idx = self.model_to_idx[true_model_id]
        mv_images = self.mv_images_h5["imgs"][idx]

        selected_numbers = random.sample(range(12), self.data_cfg.metadata.num_views)
        mv_images = mv_images[selected_numbers]
        
        return {
            "mv_images": mv_images,
            "model_id": true_model_id,
            "text": self.model_id_to_text_features[model_id]['caption'],
            "text_features": self.model_id_to_text_features[model_id]['embedding']
        }
    