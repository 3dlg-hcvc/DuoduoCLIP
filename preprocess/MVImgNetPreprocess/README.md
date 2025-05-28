## MVImgNet Data Processing

1. Download MVImgNet images from [here](https://github.com/GAP-LAB-CUHK-SZ/MVImgNet). Download all the ***mvi_\*.zip*** files (we will refer to the folder downloaded to as ***/root_path***).
2. Extract images from mvi folders (will only keep 12 images max for each object). Will create folder ***extract/class_id/object_id/\*.jpg***
```
python extract_mvimgnet.py --root_path=/root_path
```
3. Combine all object images into a h5 file. This step will also filter out all objects without at least 12 views. Will create ***data/mvimgnet_mv_images.h5*** and ***data/mvimgnet_model_to_idx.json***.
```
python create_img_h5.py
```
4. Create image and text embeddings that will be used for training. Will create ***data/mvimgnet_embeddings.h5***.
```
# Create image embeddings
python process_img_embedding.py

# Create text embeddings from text captions in captions/packed.json.gz
python process_text_embedding.py

# Combine the text and image embedding into one h5 file.
python join_text_img_embeddings.py
```
5. Get class embeddings for each class in MVImgNet for valiation. Will create ***data/mvimgnet_class_label_embeddings.npy***.
```
python gen_class_embeddings.py
```
6. Move below files into the ***dataset/data/ViT-B-32_laion2b_s34b_b79k/mvimgnet*** directory of DuoduoCLIP main path for training.
```
data/mvimgnet_mv_images.h5
data/mvimgnet_embeddings.h5
data/mvimgnet_model_to_idx.json
data/mvimgnet_class_label_embeddings.npy
```
