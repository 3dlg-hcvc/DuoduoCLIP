import os
import hydra
import torch
import numpy as np
from tqdm import tqdm
import lightning.pytorch as pl
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from huggingface_hub import hf_hub_download

from src.data.dataset.mvpnet import MVPNet
from src.model.duoduoclip import DuoduoCLIP
from src.data.data_module import _collate_fn


@torch.no_grad()
@torch.autocast(device_type="cuda")
def run_mvpnet_test_epoch(cfg, model):
    # initialize mvpnet data
    mvpnet_dataset = MVPNet(cfg.data.val, "test", "test")
    mvpnet_dataloader = DataLoader(
        mvpnet_dataset, batch_size=200, pin_memory=True,
        num_workers=cfg.data.val.dataloader.num_workers, drop_last=False, collate_fn=_collate_fn
    )

    # initialize mvpnet evaluators
    top_ks = (1, 3, 5)
    tok_k_acc = {}
    for k in top_ks:
        tok_k_acc[k] = Accuracy(task="multiclass", num_classes=cfg.data.val.evaluator.mvimgnet_num_classes, top_k=k).to(model.device)

    # run test
    for data_dict in tqdm(mvpnet_dataloader):
        data_dict['mv_images'] = data_dict['mv_images'].to(torch.float16) / 255
        for data_key in ("mv_images", "category_clip_features", "class_idx"):
            data_dict[data_key] = data_dict[data_key].to(model.device)
        output_dict = model(data_dict)

        logits = F.normalize(output_dict["mv_image_features"], dim=1) @ F.normalize(data_dict["category_clip_features"], dim=1).T

        for k in top_ks:
            tok_k_acc[k].update(logits, data_dict["class_idx"])

    # print the results
    line_width = 60
    print("{} View MVPNet LVIS Test Results (Single Run):".format(cfg.data.val.metadata.num_views))
    print('=' * line_width)
    print(" | ".join([f"{header:<5}" for header in [f"Top-{top_k}" for top_k in top_ks]]) + " |")
    print('-' * line_width)

    formatted_accuracies = ["{:<5}".format(round(tok_k_acc[k].compute().cpu().item() * 100, 2)) for k in top_ks]
    print(" | ".join(formatted_accuracies) + " |")
    print()

    results = [round(tok_k_acc[k].compute().cpu().item() * 100, 2) for k in top_ks]
    return results


@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(_cfg):
    ckpt_path = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP', filename=_cfg.ckpt_path)
    duoduoclip = DuoduoCLIP.load_from_checkpoint(ckpt_path)
    duoduoclip.eval()
    duoduoclip.cuda()

    ################################################################################################################################
    """
    Temporary for initial release and evaluation
    """
    cfg = duoduoclip.cfg

    cfg.data.val.metadata.mvimgnet_mv_data_h5 = 'dataset/data/ViT-B-32_laion2b_s34b_b79k/mvimgnet/mvimgnet_mv_images.h5'
    cfg.data.val.metadata.mvimgnet_model_to_idx = 'dataset/data/ViT-B-32_laion2b_s34b_b79k/mvimgnet/mvimgnet_model_to_idx.json'

    cfg.data.val.evaluator.mvimgnet_num_classes = 238
    cfg.data.val.metadata.mvimgnet_data_list_path = 'dataset/csv_list/mvpnet_val.csv'
    cfg.data.val.metadata.mvimgnet_clip_feat_path = 'dataset/data/ViT-B-32_laion2b_s34b_b79k/mvimgnet/mvimgnet_class_label_embeddings.npy'
    ################################################################################################################################

    all_results = {}

    for num_views in [1, 6, 12]:
        cfg.data.val.metadata.num_views = num_views

        pl.seed_everything(cfg.test_seed + 555, workers=True)
        results_1 = run_mvpnet_test_epoch(cfg, duoduoclip)

        pl.seed_everything(cfg.test_seed + 666, workers=True)
        results_2 = run_mvpnet_test_epoch(cfg, duoduoclip)

        pl.seed_everything(cfg.test_seed + 777, workers=True)
        results_3 = run_mvpnet_test_epoch(cfg, duoduoclip)

        results = np.array([results_1, results_2, results_3])
        results = results.mean(0)
        results = results.round(decimals=2)

        all_results[num_views] = results

    # print the results
    line_width = 60
    print('=' * line_width)
    print("All Views MVPNet Test Results (3 Runs):")
    print('=' * line_width)
    print("{:<5}".format("Views") + " | " + " | ".join([f"{header:<5}" for header in [f"Top-{top_k}" for top_k in [1, 3, 5]]]) + " |")
    print('-' * line_width)

    for num_views in [1, 6, 12]:
        formatted_accuracies = ["{:<5}".format(all_results[num_views][k]) for k in [0, 1, 2]]
        print("{:<5}".format(num_views) + " | " + " | ".join(formatted_accuracies) + " |")
        print('=' * line_width)


if __name__ == '__main__':
    main()
