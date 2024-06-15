import torch
import numpy as np
from tqdm import tqdm
from functools import partial
import lightning.pytorch as pl
from importlib import import_module
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from .dataset import mvimgnet

class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, model_network):
        super().__init__()
        self.data_cfg = data_cfg
        val_dataset_name = data_cfg.val.dataset
        train_dataset_name = data_cfg.train.dataset

        self.val_dataset = getattr(
            import_module(f"src.data.dataset.{val_dataset_name.lower()}"), val_dataset_name
        )
        self.train_dataset = getattr(
            import_module(f"src.data.dataset.{train_dataset_name.lower()}"), train_dataset_name
        )

        self.model_name = '{}_{}'.format(model_network.base_name, model_network.pretrained_name)
        self.data_info_train = self._read_data_info_train()
        self._load_text_data_train()

    def _read_data_info_train(self):
        data_info = []
        with open(self.data_cfg.train.metadata.data_list_path, "r") as f:
            model_ids = f.read().splitlines()
        data_info = [{'model_id': model_id} for model_id in model_ids]
        return data_info

    def _load_text_data_train(self):
        print('Loading text data npy')
        text_data = np.load(self.data_cfg.train.metadata.text_data, allow_pickle=True).item()

        self.text_data_train = {}
        for i, data in enumerate(tqdm(self.data_info_train, desc="Loading text data into memory")):
            self.text_data_train[i] = text_data[f"{data['model_id']}"]

    def setup(self, stage=None):
        if stage == "fit":
            self.train_set = self.train_dataset(self.data_cfg.train, "train", stage, self.model_name, self.data_info_train, self.text_data_train)

            if self.data_cfg.train.metadata.use_mvimgnet:
                self.mvimgnet_set = mvimgnet.MVImgNet(self.data_cfg.train, "train", stage)

            self.val_set = self.val_dataset(self.data_cfg.val, "val", stage, self.model_name)

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_set, batch_size=self.data_cfg.train.dataloader.batch_size, shuffle=True,
            pin_memory=True, num_workers=self.data_cfg.train.dataloader.num_workers, drop_last=True,
            collate_fn=partial(_train_collate_fn, self.train_set),
            persistent_workers=not (self.data_cfg.train.dataloader.num_workers == 0),
        )

        if self.data_cfg.train.metadata.use_mvimgnet:
            dataset_fraction = self.data_cfg.train.metadata.mvimgnet_fraction
            mvimgnet_dataloader = DataLoader(
                self.mvimgnet_set, batch_size=self.data_cfg.train.dataloader.batch_size // dataset_fraction, shuffle=True,
                pin_memory=True, num_workers=self.data_cfg.train.dataloader.num_workers, drop_last=True,
                collate_fn=partial(_mvimgnet_train_collate_fn, self.mvimgnet_set),
                persistent_workers=not (self.data_cfg.train.dataloader.num_workers == 0),
            )
            loaders = {"objaverse": train_dataloader, "mvimgnet": mvimgnet_dataloader}
            return loaders
        else:
            return train_dataloader

    def val_dataloader(self):
        return DataLoader(
            self.val_set, batch_size=self.data_cfg.val.dataloader.batch_size, pin_memory=True,
            num_workers=self.data_cfg.val.dataloader.num_workers, collate_fn=_collate_fn,
            persistent_workers=True,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        if type(batch) == tuple:
            batch = batch[0]

        if 'mvimgnet' in batch.keys():
            batch['mvimgnet']['mv_images'] = batch['mvimgnet']['mv_images'].to(torch.float16) / 255
            batch['objaverse']['mv_images'] = batch['objaverse']['mv_images'].to(torch.float16) / 255
        else:
            batch['mv_images'] = batch['mv_images'].to(torch.float16) / 255

        return batch


def _collate_fn(batch):
    default_collate_items = ("mv_images", "class_idx")
    batch_data = []
    for _, b in enumerate(batch):
        batch_data.append({k: b[k] for k in default_collate_items})
    data_dict = default_collate(batch_data)
    data_dict["category_clip_features"] = torch.from_numpy(batch[0]["category_clip_features"])
    return data_dict


def _train_collate_fn(dataset_input, batch):
    dataset_input.get_selected_numbers()

    default_collate_items = ("text_features", "model_id")
    batch_data = []
    mv_images = np.empty(shape=(len(batch), *batch[0]['mv_images'].shape), dtype=batch[0]['mv_images'].dtype)
    image_features = np.empty(shape=(len(batch), *batch[0]['image_features'].shape), dtype=batch[0]['image_features'].dtype)
    for i, b in enumerate(batch):
        batch_data.append({k: b[k] for k in default_collate_items})

        mv_images[i] = b['mv_images']
        image_features[i] = b['image_features']
    data_dict = default_collate(batch_data)

    image_features = image_features.mean(1)
    data_dict['mv_images'] = torch.from_numpy(mv_images)
    data_dict['image_features'] = torch.from_numpy(image_features)

    return data_dict

def _mvimgnet_train_collate_fn(dataset_input, batch):
    dataset_input.get_selected_numbers()

    mv_images = np.empty(shape=(len(batch), *batch[0]['mv_images'].shape), dtype=batch[0]['mv_images'].dtype)
    text_features = np.empty(shape=(len(batch), *batch[0]['text_features'].shape), dtype=batch[0]['image_features'].dtype)
    image_features = np.empty(shape=(len(batch), *batch[0]['image_features'].shape), dtype=batch[0]['image_features'].dtype)
    for i, b in enumerate(batch):
        mv_images[i] = b['mv_images']
        text_features[i] = b['text_features']
        image_features[i] = b['image_features']

    text_features = text_features.mean(1)
    image_features = image_features.mean(1)

    data_dict = {}
    data_dict['mv_images'] = torch.from_numpy(mv_images)
    data_dict['text_features'] = torch.from_numpy(text_features)
    data_dict['image_features'] = torch.from_numpy(image_features)

    return data_dict
