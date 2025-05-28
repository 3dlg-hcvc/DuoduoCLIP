import os
import glob
import shutil
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Arguments")
parser.add_argument("--root_path", type=str, required=True)
args = parser.parse_args()

mvi_folders = glob.glob(os.path.join(args.root_path, 'mvi_*'))

for mvi_folder in mvi_folders:
    cat_folders = glob.glob(os.path.join(mvi_folder, '*'))
    for cat_folder in tqdm(cat_folders):
        obj_folders = glob.glob(os.path.join(cat_folder, '*'))

        for obj_folder in obj_folders:
            image_files = sorted(glob.glob(os.path.join(obj_folder, 'images', '*.jpg')))

            if len(image_files) > 12:
                extract_image_files = image_files[::len(image_files)//12][:12]
            else:
                extract_image_files = image_files

            write_path = os.path.join('extract', obj_folder.split('/')[-2], obj_folder.split('/')[-1])
            if not os.path.exists(write_path):
                os.makedirs(write_path)

            for counter, extract_image_file in enumerate(extract_image_files):
                formatted_number = "%02d" % counter
                shutil.copy(extract_image_files[counter], os.path.join(write_path, '{}.jpg'.format(formatted_number)))
