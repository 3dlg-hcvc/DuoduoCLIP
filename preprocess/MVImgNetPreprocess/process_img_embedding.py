import h5py
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class MVImagesDataset(Dataset):
    def __init__(self, h5file, preprocess):
        self.h5file = h5py.File(h5file, "r")
        self.preprocess = preprocess

    def __len__(self):
        return len(self.h5file['imgs'])
    
    def _process_image_data(self, idx):
        mv_images_raw = self.h5file['imgs'][idx]
        mv_images_processed = torch.empty(size=mv_images_raw.shape, dtype=torch.float16)
        for i, single_view_img in enumerate(mv_images_raw):
            mv_images_processed[i] = self.preprocess(Image.fromarray(single_view_img.transpose(1, 2, 0), mode="RGB")).to(torch.float16)
        return mv_images_processed

    def __getitem__(self, idx):
        img = self._process_image_data(idx)
        return {'images': img, 'idx': idx}
    
output_path = 'data/mvimgnet_image_embeddings.h5'
open_clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda', precision="fp16")
    
batch_size = 10
dataset = MVImagesDataset(h5file='data/mvimgnet_mv_images.h5', preprocess=preprocess)
mv_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

@torch.no_grad()
@torch.autocast(device_type="cuda")
def generate_clip_image_embeddings(clip_model, batch_imgs):
    return clip_model.encode_image(batch_imgs)

with h5py.File(output_path, "w") as h5_file:
    h5_file.create_dataset("img_feat", shape=(len(dataset), 12, open_clip_model.visual.output_dim), dtype=np.float16)

    for i, item in enumerate(tqdm(mv_dataloader, desc="Processing image embeddings")):
        ids = item['idx']
        images = item['images']
        local_batch_size = images.shape[0]
        img_embeddings = generate_clip_image_embeddings(open_clip_model, images.flatten(0, 1).cuda()).cpu().numpy()

        h5_file["img_feat"][i * batch_size:i * batch_size + local_batch_size] = img_embeddings.reshape(local_batch_size, -1, open_clip_model.visual.output_dim)
