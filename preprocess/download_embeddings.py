import os
import shutil
from huggingface_hub import hf_hub_download

output_folder = 'dataset/data/objaverse_embeddings/Four_1to6F_bs1600_LT6'
os.makedirs(output_folder, exist_ok=True)

# Download files from huggingface hub
shape_emb_objaverse = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='model_embeddings/Four_1to6F_bs1600_LT6/shape_emb_objaverse.h5', repo_type="dataset")
shape_emb_objaverse_model_to_idx = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='model_embeddings/Four_1to6F_bs1600_LT6/shape_emb_objaverse_model_to_idx.json', repo_type="dataset")

# Move files to dataset/data
shutil.move(os.path.realpath(shape_emb_objaverse), os.path.join(output_folder, 'shape_emb_objaverse.h5'))
shutil.move(os.path.realpath(shape_emb_objaverse_model_to_idx), os.path.join(output_folder, 'shape_emb_objaverse_model_to_idx.json'))
