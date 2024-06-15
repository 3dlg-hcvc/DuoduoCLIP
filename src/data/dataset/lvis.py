import csv
import json
import random
import numpy as np
from .general_dataset import GeneralDataset


class LVIS(GeneralDataset):
    def __init__(self, data_cfg, split, stage, model_name):
        super().__init__(data_cfg, split, stage, model_name)

        self.data_info = self._read_data_info()
        self.clip_cat_feat = np.load(self.data_cfg.metadata.clip_feat_path, allow_pickle=True)
        self.num_views = self.data_cfg.metadata.num_views

        with open(self.data_cfg.metadata.model_to_idx, 'r') as file:
            self.model_to_idx = json.load(file)

    def __getitem__(self, idx):
        super().__getitem__(idx)

        model_id = self.data_info[idx]["model_id"]
        mv_images = self._process_mv_data(model_id)

        return {
            "mv_images": mv_images,
            "class_idx": self.data_info[idx]["class_idx"],
            "category_clip_features": self.clip_cat_feat
        }

    def _read_data_info(self):
        data_info = []
        with open(self.data_cfg.metadata.data_list_path, "r") as f:
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

        return mv_images
