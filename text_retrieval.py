import json
import h5py
import faiss
import hydra
import torch
import numpy as np
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from src.model.duoduoclip import DuoduoCLIP


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(_cfg):
    ckpt_path = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP', filename=_cfg.ckpt_path)
    duoduoclip = DuoduoCLIP.load_from_checkpoint(ckpt_path)
    duoduoclip.eval()
    duoduoclip.cuda()

    # Get embeddings for objaverse objects
    shape_embeddings_h5 =  h5py.File('dataset/data/objaverse_embeddings/Four_1to6F_bs1600_LT6/shape_emb_objaverse.h5', "r")
    shape_embeddings = shape_embeddings_h5['shape_feat'][:]
    shape_embeddings_h5.close()
    shape_embeddings = shape_embeddings.astype(np.float32)

    # Normalize the shape embeddings
    shape_embeddings = torch.from_numpy(shape_embeddings).cuda()
    shape_embeddings = F.normalize(shape_embeddings, dim=1)
    shape_embeddings = shape_embeddings.cpu().numpy()

    # Get index of faiss
    index = faiss.IndexFlatIP(512)
    index.add(shape_embeddings)

    # Load mapping for shape embeddings in search library
    with open('dataset/data/objaverse_embeddings/Four_1to6F_bs1600_LT6/shape_emb_objaverse_model_to_idx.json', 'r') as file:
        shape_model_to_idx = json.load(file)

    # Get reverse mappings
    shape_idx_to_model = {}
    for key, value in shape_model_to_idx.items():
        shape_idx_to_model[value] = key

    while True:
        line_width = 60
        print('=' * line_width)

        input_prompt = input("Enter a description to search models by (or 'exit' to quit): ")
        if input_prompt.lower() == 'exit':
            print("Exiting...")
            break

        text = duoduoclip.tokenizer([input_prompt]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = duoduoclip.duoduoclip.encode_text(text)
            text_features = F.normalize(text_features, dim=1)

        query_emb = F.normalize(text_features, dim=1)
        query_emb = query_emb.cpu().numpy().astype(np.float32)
        _, I = index.search(query_emb, 5)

        print()
        print('Top 5 retrieved models:')
        for i in range(5):
            print(shape_idx_to_model[I[0][i]])
        print('=' * line_width)
        print()


if __name__ == '__main__':
    main()
