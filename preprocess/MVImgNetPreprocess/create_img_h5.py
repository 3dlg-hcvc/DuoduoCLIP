import os
import glob
import h5py
import json
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class MVImagesProcessingDataset(Dataset):
    def __init__(self, filtered_obj_folders):
        self.filtered_obj_folders = filtered_obj_folders

    def __len__(self):
        return len(self.filtered_obj_folders)

    def __getitem__(self, idx):
        obj_folder = self.filtered_obj_folders[idx]
        model_id = '_'.join(obj_folder.split('/')[-2:])
        small_images_raw = np.empty(shape=(12, 3, 224, 224), dtype=np.uint8)

        img_files = glob.glob(os.path.join(obj_folder, '*.jpg'))
        for counter_mv, img_file in enumerate(img_files):
            orig_img = Image.open(img_file)
            resized_image = orig_img.resize((224, 224))
            resized_image = np.asarray(resized_image).transpose(2, 0, 1)
            small_images_raw[counter_mv] = resized_image
        
        return {'model_ids': model_id, 'small_images_raw': small_images_raw}

os.makedirs('data', exist_ok=True)
output_path = 'data/mvimgnet_mv_images.h5'
root_path = './extract'
cat_folders = glob.glob(os.path.join(root_path, '*'))

'''
Get all object folders
'''
obj_folders = []
for cat_folder in cat_folders:
    cat_obj_folders = glob.glob(os.path.join(cat_folder, '*'))
    obj_folders += cat_obj_folders

'''
Get objects only with number of images larger than 12
'''
filtered_obj_folders = []
for obj_folder in tqdm(obj_folders):
        img_files = glob.glob(os.path.join(obj_folder, '*.jpg'))
        if len(img_files) >= 12:
             filtered_obj_folders.append(obj_folder)

'''
Create the dataset
'''
batch_size = 2
dataset = MVImagesProcessingDataset(filtered_obj_folders=filtered_obj_folders)
mv_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True,num_workers=4, drop_last=False)

'''
Create h5 of mvimages
'''
model_to_idx = {}
with h5py.File(output_path, "w") as f:
    f.create_dataset("imgs", shape=(len(filtered_obj_folders), 12, 3, 224, 224), dtype=np.uint8)

    for i, item in enumerate(tqdm(mv_dataloader, desc="Processing multi-view images")):
        model_ids = item['model_ids']
        mv_images = item['small_images_raw']
        current_batch_start_idx = i * batch_size
        f["imgs"][current_batch_start_idx:current_batch_start_idx+mv_images.shape[0]] = mv_images.numpy()

        for counter, model_id in enumerate(model_ids):
            model_to_idx[model_id] = current_batch_start_idx + counter

with open('data/mvimgnet_model_to_idx.json', 'w') as file:
    json.dump(model_to_idx, file)
