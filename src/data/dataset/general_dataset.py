import h5py
from torch.utils.data import Dataset


class GeneralDataset(Dataset):
    def __init__(self, data_cfg, split, stage, model_name):
        self.data_cfg = data_cfg
        self.split = split
        self.stage = stage
        self.model_name = model_name

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        if not hasattr(self, 'mv_images_h5'):
            self.mv_images_h5 = h5py.File(self.data_cfg.metadata.mv_data_h5, "r")

        if not hasattr(self, 'vision_data_h5') and self.split == 'train':
            self.vision_data_h5 = h5py.File(self.data_cfg.metadata.vision_data, "r")
