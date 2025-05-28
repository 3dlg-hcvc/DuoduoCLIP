import h5py
import numpy as np

output_path = "data/mvimgnet_embeddings.h5"
img_emb_h5 = h5py.File("data/mvimgnet_image_embeddings.h5", "r")
text_emb_h5 = h5py.File("data/mvimgnet_text_embeddings.h5", "r")

assert len(img_emb_h5['img_feat']) == len(text_emb_h5['text_feat'])
num_model_ids = len(img_emb_h5['img_feat'])

_, emb_dim = img_emb_h5['img_feat'][0].shape

with h5py.File(output_path, "w") as h5_file:
    h5_file.create_dataset("img_feat", shape=(num_model_ids, 12, emb_dim), dtype=np.float16)
    h5_file.create_dataset("text_feat", shape=(num_model_ids, 12, emb_dim), dtype=np.float16)

    h5_file['img_feat'][:] = img_emb_h5['img_feat'][:]
    h5_file['text_feat'][:] = text_emb_h5['text_feat'][:]
