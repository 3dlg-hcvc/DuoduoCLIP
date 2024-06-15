import os
import shutil
import subprocess
from huggingface_hub import hf_hub_download

output_folder = 'dataset/data'
os.makedirs(output_folder, exist_ok=True)

# Download files from huggingface hub
model_to_idx = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='lvis_split/lvis_model_to_idx.json', repo_type="dataset")
lvis_split_1 = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='lvis_split/lvis_images.h5_part_aa', repo_type="dataset")
lvis_split_2 = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='lvis_split/lvis_images.h5_part_ab', repo_type="dataset")
lvis_class_label_embeddings = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP-data', filename='lvis_split/lvis_class_label_embeddings.npy', repo_type="dataset")

# Move files to dataset/data
shutil.move(os.path.realpath(model_to_idx), os.path.join(output_folder, 'lvis_model_to_idx.json'))
shutil.move(os.path.realpath(lvis_split_1), os.path.join(output_folder, 'lvis_images.h5_part_aa'))
shutil.move(os.path.realpath(lvis_split_2), os.path.join(output_folder, 'lvis_images.h5_part_ab'))
shutil.move(os.path.realpath(lvis_class_label_embeddings), os.path.join(output_folder, 'lvis_class_label_embeddings.npy'))

# Merge to make complete lvis h5 file
dest = os.path.join(output_folder, 'lvis_images.h5')
src = os.path.join(output_folder, 'lvis_images.h5_part_*')
command = "cat {} > {}".format(src, dest)
result = subprocess.run(command, shell=True, check=True)

# Delete the split files
# os.remove(os.path.join(output_folder, 'lvis_images.h5_part_aa'))
# os.remove(os.path.join(output_folder, 'lvis_images.h5_part_ab'))
