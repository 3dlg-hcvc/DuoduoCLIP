import json
import h5py
import numpy as np
from torch.utils.data import Dataset

class MVImgNet(Dataset):
    def __init__(self, data_cfg, split, stage):
        self.stage = stage
        self.split = split
        self.data_cfg = data_cfg
        self.num_views = self.data_cfg.metadata.num_views

        with open(self.data_cfg.metadata.mvimgnet_model_to_idx, 'r') as file:
            self.mvimgnet_model_to_idx = json.load(file)

        if data_cfg.metadata.mvimgnet_mode == 'mvimgnet':
            with open(self.data_cfg.metadata.mvimgnet_list, "r") as f:
                self.model_ids = f.read().splitlines()
        elif data_cfg.metadata.mvimgnet_mode == 'mvpnet':
            with open(self.data_cfg.metadata.mvpnet_list, "r") as f:
                self.model_ids = f.read().splitlines()
        else:
            raise NotImplementedError

        self.mvimgnet_emb_h5 = h5py.File(self.data_cfg.metadata.mvimgnet_emb_h5, "r")
        self.mvimgnet_mv_data_h5 = h5py.File(self.data_cfg.metadata.mvimgnet_mv_data_h5, "r")

        self.get_selected_numbers()

    def get_selected_numbers(self):
        low, high = self.num_views
        high = high + 1
        num_frames = np.random.randint(low=low, high=high)
        self.selected_numbers = np.random.choice(12, size=num_frames, replace=False)
        self.selected_numbers.sort()

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        model_id = self.model_ids[idx]
        img = self.mvimgnet_mv_data_h5['imgs'][self.mvimgnet_model_to_idx[model_id], self.selected_numbers]
        text_feats = self.mvimgnet_emb_h5["text_feat"][self.mvimgnet_model_to_idx[model_id], self.selected_numbers]
        image_feats = self.mvimgnet_emb_h5["img_feat"][self.mvimgnet_model_to_idx[model_id], self.selected_numbers]

        return {
            "mv_images": img,
            "text_features": text_feats,
            "image_features": image_feats,
        }
