import json
import h5py
import gzip
import torch
import open_clip
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

class MVTextDataset(Dataset):
    def __init__(self, text_file, model_ids):
        with gzip.open(text_file, 'rt', encoding='utf-8') as f:
            self.model_text = json.load(f)
        self.model_ids = model_ids

    def __len__(self):
        return len(self.model_ids)

    def __getitem__(self, idx):
        model_id = self.model_ids[idx]
        text = self.model_text[model_id]
        return text
    
def collate_fn(batch):
    all_text = []
    for i, b in enumerate(batch):
        all_text.append(b)
    return all_text
    
with open('./data/mvimgnet_model_to_idx.json', "r") as f:
    model_to_idx = json.load(f)
model_ids = list(model_to_idx.keys())

sorted = [i for i in range(len(model_to_idx))]
check_sorted = [model_to_idx[model_id] for model_id in model_ids]
assert sorted == check_sorted

output_path = 'data/mvimgnet_text_embeddings.h5'
open_clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device='cuda', precision="fp16")
tokenizer = open_clip.get_tokenizer('ViT-B-32')

batch_size = 10
dataset = MVTextDataset(text_file='./captions/packed.json.gz', model_ids=model_ids)
text_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4, drop_last=False, collate_fn=collate_fn)

@torch.no_grad()
@torch.autocast(device_type="cuda")
def generate_clip_text_embeddings(clip_model, batch_texts, tokenizer, device):
    batch_tokens = tokenizer(batch_texts).to(device)
    return clip_model.encode_text(batch_tokens).cpu().numpy()

with h5py.File(output_path, "w") as h5_file:
    h5_file.create_dataset("text_feat", shape=(len(dataset), 12, open_clip_model.visual.output_dim), dtype=np.float16)

    for i, text in enumerate(tqdm(text_dataloader, desc="Processing text embeddings")):
        all_text = []
        for sublist in text:
            all_text.extend(sublist)

        local_batch_size = len(text)
        text_embeddings = generate_clip_text_embeddings(open_clip_model, all_text, tokenizer, device='cuda')
        h5_file["text_feat"][i * batch_size:i * batch_size + local_batch_size] = text_embeddings.reshape(local_batch_size, -1, open_clip_model.visual.output_dim)
