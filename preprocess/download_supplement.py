import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download

output_folder = 'dataset/data'
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'ViT-B-32_laion2b_s34b_b79k'), exist_ok=True)

# Download files from huggingface hub
model_to_idx = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='training/four/model_to_idx.json', repo_type="dataset")
supplement_split_1 = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='training/four/supplement_mv_images_part_aa', repo_type="dataset")
supplement_split_2 = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='training/four/supplement_mv_images_part_ab', repo_type="dataset")
supplement_split_3 = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='training/four/supplement_mv_images_part_ac', repo_type="dataset")
supplement_model_to_idx = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='training/four/supplement_model_to_idx.json', repo_type="dataset")
lvis_class_label_embeddings = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='lvis_split/lvis_class_label_embeddings.npy', repo_type="dataset")
text_embeddings = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='training/four/ViT-B-32_laion2b_s34b_b79k/text_embeddings.npy', repo_type="dataset")
image_embeddings = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='training/four/ViT-B-32_laion2b_s34b_b79k/image_embeddings.h5', repo_type="dataset")

# Move files to dataset/data
shutil.move(os.path.realpath(model_to_idx), os.path.join(output_folder, 'model_to_idx.json'))
shutil.move(os.path.realpath(supplement_split_1), os.path.join(output_folder, 'supplement_mv_images_part_aa'))
shutil.move(os.path.realpath(supplement_split_2), os.path.join(output_folder, 'supplement_mv_images_part_ab'))
shutil.move(os.path.realpath(supplement_split_3), os.path.join(output_folder, 'supplement_mv_images_part_ac'))
shutil.move(os.path.realpath(supplement_model_to_idx), os.path.join(output_folder, 'supplement_model_to_idx.json'))
shutil.move(os.path.realpath(text_embeddings), os.path.join(output_folder, 'ViT-B-32_laion2b_s34b_b79k', 'text_embeddings.npy'))
shutil.move(os.path.realpath(image_embeddings), os.path.join(output_folder, 'ViT-B-32_laion2b_s34b_b79k', 'image_embeddings.h5'))
shutil.move(os.path.realpath(lvis_class_label_embeddings), os.path.join(output_folder, 'ViT-B-32_laion2b_s34b_b79k', 'lvis_class_label_embeddings.npy'))

# Merge to make complete supplement h5 file
dest = os.path.join(output_folder, 'supplement_mv_images.h5')
src = os.path.join(output_folder, 'supplement_mv_images_part_*')
command = "cat {} > {}".format(src, dest)
result = subprocess.run(command, shell=True, check=True)
