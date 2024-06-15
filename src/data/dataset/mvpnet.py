import csv
import json
import h5py
import random
import numpy as np
from torch.utils.data import Dataset

class MVPNet(Dataset):
    def __init__(self, data_cfg, split, stage):
        self.stage = stage
        self.split = split
        self.data_cfg = data_cfg
        self.num_views = self.data_cfg.metadata.num_views

        self.data_info = self._read_data_info()
        with open(self.data_cfg.metadata.mvimgnet_model_to_idx, 'r') as file:
            self.mvimgnet_model_to_idx = json.load(file)
        self.mvimgnet_mv_data_h5 = h5py.File(self.data_cfg.metadata.mvimgnet_mv_data_h5, "r")
        self.clip_cat_feat = np.load(self.data_cfg.metadata.mvimgnet_clip_feat_path, allow_pickle=True)

    def _read_data_info(self):
        data_info = []
        with open(self.data_cfg.metadata.mvimgnet_data_list_path, "r") as f:
            csv_reader = csv.reader(f)
            for line in csv_reader:
                data_info.append(
                    {
                        "class_idx": int(line[0]),
                        "model_id": line[-1],
                    }
                )
        return data_info

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        model_id = self.data_info[idx]["model_id"]
        image_idx = self.mvimgnet_model_to_idx[model_id]
        mv_images = self.mvimgnet_mv_data_h5["imgs"][image_idx]

        selected_numbers = random.sample(range(12), self.num_views)
        mv_images = mv_images[selected_numbers]

        return {
            "model_id": model_id,
            "mv_images": mv_images,
            "class_idx": self.data_info[idx]["class_idx"],
            "category_clip_features": self.clip_cat_feat
        }
