import json
import numpy as np
from .general_dataset import GeneralDataset


class Four(GeneralDataset):
    def __init__(self, data_cfg, split, stage, model_name, data_info, text_data):
        super().__init__(data_cfg, split, stage, model_name)

        self.data_info = data_info
        self.text_data = text_data
        self.num_views = self.data_cfg.metadata.num_views

        with open(self.data_cfg.metadata.model_to_idx, 'r') as file:
            self.model_to_idx = json.load(file)

        self.get_selected_numbers()

    def get_selected_numbers(self):
        low, high = self.num_views
        high = high + 1
        num_frames = np.random.randint(low=low, high=high)
        self.selected_numbers = np.random.choice(12, size=num_frames, replace=False)
        self.selected_numbers.sort()

    def __getitem__(self, idx):
        super().__getitem__(idx)

        model_id = self.data_info[idx]["model_id"]
        mv_images = self._process_mv_data(model_id)
        text_feats = self._process_text_data(idx)
        image_feats = self._process_image_data(model_id)

        return {
            "model_id": model_id,
            "mv_images": mv_images,
            "text_features": text_feats,
            "image_features": image_feats,
            "num_views": self.num_views,
        }

    def _process_text_data(self, idx):
        data = self.text_data[idx]
        text_feat = []
        if "text_feat" in data:
            idx = np.random.randint(len(data["text_feat"]))
            text_feat.append(data["text_feat"][idx])

        if np.random.rand() < 0.5:
            if "blip_caption_feat" in data:
                text_feat.append(data["blip_caption_feat"])
        else:
            if "msft_caption_feat" in data:
                text_feat.append(data["msft_caption_feat"])

        idx = np.random.randint(len(data["retrieval_text_feat"]))
        text_feat.append(data["retrieval_text_feat"][idx])

        text_idx = np.random.randint(len(text_feat))
        text_feat = text_feat[text_idx]

        return text_feat

    def _process_image_data(self, model_id):
        return self.vision_data_h5["img_feat"][self.model_to_idx[model_id], self.selected_numbers]

    def _process_mv_data(self, model_id):
        return self.mv_images_h5["imgs"][self.model_to_idx[model_id], self.selected_numbers]
