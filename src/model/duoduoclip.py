import gc
import hydra
import lightning.pytorch as pl
from omegaconf import OmegaConf

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .. import custom_clip

def get_model(base_name='ViT-L-14', pretrained_name='laion2b_s32b_b82k'):
    tokenizer = custom_clip.get_tokenizer(base_name)

    model_pretrained, _, _ = custom_clip.create_model_and_transforms(base_name, pretrained=pretrained_name)
    model, _, _ = custom_clip.create_model_and_transforms(base_name + '-MV')
    model.load_state_dict(model_pretrained.state_dict(), strict=True)

    del model_pretrained
    gc.collect()
    torch.cuda.empty_cache()

    return model, tokenizer

class DuoduoCLIP(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        
        if type(cfg) == dict:
            cfg = OmegaConf.create(cfg)
        self.save_hyperparameters(OmegaConf.to_container(cfg))

        self.cfg = cfg
        self.layers_threshold = cfg.model.network.layers_threshold
        self.contrastive_loss = hydra.utils.instantiate(cfg.model.loss.flag)
        self.duoduoclip, self.tokenizer = get_model(cfg.model.network.base_name, cfg.model.network.pretrained_name)

        self.unlock_mha()

        self.train_norm_transform = transforms.Compose([
            transforms.RandomCrop((224 - 16, 224 - 16)),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self.val_norm_transform = transforms.Compose([
            transforms.RandomCrop((224 - 16, 224 - 16)),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        self.lambda_text = cfg.model.network.lambda_text
        self.lambda_image = cfg.model.network.lambda_image

        self.objaverse_lvis_acc_top_1 = hydra.utils.instantiate(cfg.data.val.evaluator, top_k=1)
        self.objaverse_lvis_acc_top_3 = hydra.utils.instantiate(cfg.data.val.evaluator, top_k=3)
        self.objaverse_lvis_acc_top_5 = hydra.utils.instantiate(cfg.data.val.evaluator, top_k=5)

    def unlock_mha(self):
        for param in self.duoduoclip.parameters():
            param.requires_grad = False
        self.duoduoclip.visual.unlock_mha(layers_threshold=self.layers_threshold)

    def encode_text(self, input_prompt):
        text = self.tokenizer([input_prompt]).cuda()
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.duoduoclip.encode_text(text)
            text_features = F.normalize(text_features, dim=1)
        return text_features
    
    def encode_image(self, mv_images):
        # Single-view image
        if len(mv_images.shape) == 3:
            mv_images = torch.from_numpy(mv_images)[None, None, ...]
        # Multi-view image
        elif len(mv_images.shape) == 4:
            mv_images = torch.from_numpy(mv_images).unsqueeze(0)
        else:
            raise NotImplementedError
        
        mv_images = mv_images.to(torch.float16).permute(0, 1, 4, 2, 3) / 255
        if mv_images.shape[3] != 224 or mv_images.shape[4] != 224:
            mv_images = F.interpolate(mv_images, size=224, mode='bilinear', align_corners=False)

        data_dict = {}
        data_dict['mv_images'] = mv_images.to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast():
            mv_image_features = self(data_dict, is_training=False)["mv_image_features"]
            mv_image_features = F.normalize(mv_image_features, dim=1)
        return mv_image_features

    def forward(self, data_dict, is_training=True):
        if is_training:
            norm_transform = self.train_norm_transform
        else:
            norm_transform = self.val_norm_transform

        if 'mvimgnet' in data_dict.keys():
            ##### Objaverse #####
            bs, f, c, h, w = data_dict['objaverse']['mv_images'].shape
            mv_images = data_dict['objaverse']['mv_images'].reshape(bs * f, c, h, w)
            mv_images = norm_transform(mv_images)

            num_frames_list = [1] * self.layers_threshold + [f] * (self.duoduoclip.visual.transformer.layers - self.layers_threshold) + [f]

            mv_image_features = self.duoduoclip.encode_image(mv_images, num_frames=num_frames_list)

            ##### MVImgNet #####
            bs, f, c, h, w = data_dict['mvimgnet']['mv_images'].shape
            mvimgnet_mv_images = data_dict['mvimgnet']['mv_images'].reshape(bs * f, c, h, w)
            mvimgnet_mv_images = norm_transform(mvimgnet_mv_images)

            num_frames_list = [1] * self.layers_threshold + [f] * (self.duoduoclip.visual.transformer.layers - self.layers_threshold) + [f]

            mvimgnet_mv_image_features = self.duoduoclip.encode_image(mvimgnet_mv_images, num_frames=num_frames_list)

            mv_image_features = torch.cat((mv_image_features, mvimgnet_mv_image_features), dim=0)

            output_dict = {"mv_image_features":  mv_image_features}

            data_dict["text_features"] = torch.cat((data_dict["objaverse"]["text_features"], data_dict["mvimgnet"]["text_features"]), dim=0)
            data_dict["image_features"] = torch.cat((data_dict["objaverse"]["image_features"], data_dict["mvimgnet"]["image_features"]), dim=0)
        else:
            bs, f, c, h, w = data_dict['mv_images'].shape
            mv_images = data_dict['mv_images'].reshape(bs * f, c, h, w)
            mv_images = norm_transform(mv_images)

            num_frames_list = [1] * self.layers_threshold + [f] * (self.duoduoclip.visual.transformer.layers - self.layers_threshold) + [f]
            
            mv_image_features = self.duoduoclip.encode_image(mv_images, num_frames=num_frames_list)

            output_dict = {"mv_image_features":  mv_image_features}

        return output_dict

    def _loss(self, data_dict, output_dict, loss_prefix):
        if self.cfg.trainer.devices > 1:
            data_dict["text_features"] = torch.cat(torch.distributed.nn.all_gather(data_dict["text_features"]), dim=0)
            data_dict["image_features"] = torch.cat(torch.distributed.nn.all_gather(data_dict["image_features"]), dim=0)
            output_dict["mv_image_features"] = torch.cat(torch.distributed.nn.all_gather(output_dict["mv_image_features"]), dim=0)

        loss_dict = {
            f"{loss_prefix}/mv_image_loss": self.lambda_image * self.contrastive_loss(
                output_dict["mv_image_features"], data_dict["image_features"]
            ),
            f"{loss_prefix}/mv_text_loss": self.lambda_text * self.contrastive_loss(
                output_dict["mv_image_features"], data_dict["text_features"]
            ),
        }

        loss_dict[f"{loss_prefix}/total_loss"] = sum(loss_dict.values())
        return loss_dict

    def configure_optimizers(self):
        optimizer = hydra.utils.instantiate(self.cfg.model.optimizer, params=self.parameters())
        scheduler = hydra.utils.instantiate(self.cfg.model.scheduler, optimizer=optimizer)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "name": "cosine_annealing_lr"}}

    def training_step(self, data_dict, idx):
        if type(data_dict) == tuple:
            data_dict = data_dict[0]
        output_dict = self(data_dict)
        loss_dict = self._loss(data_dict, output_dict, "train_loss")

        # calculate the total loss and log
        self.log_dict(loss_dict, on_step=True, on_epoch=False)
        return loss_dict["train_loss/total_loss"]

    def validation_step(self, data_dict, idx):
        output_dict = self(data_dict, is_training=False)
        category_clip_features = data_dict["category_clip_features"]
        logits = F.normalize(output_dict["mv_image_features"], dim=1) @ F.normalize(category_clip_features, dim=1).T

        self.objaverse_lvis_acc_top_1.update(logits, data_dict["class_idx"])
        self.objaverse_lvis_acc_top_3.update(logits, data_dict["class_idx"])
        self.objaverse_lvis_acc_top_5.update(logits, data_dict["class_idx"])

    def on_validation_epoch_end(self):
        acc = self.objaverse_lvis_acc_top_1.compute()
        self.log(f"val_eval/objaverse_lvis_acc_top_1", acc, sync_dist=True)
        self.objaverse_lvis_acc_top_1.reset()

        acc = self.objaverse_lvis_acc_top_3.compute()
        self.log(f"val_eval/objaverse_lvis_acc_top_3", acc, sync_dist=True)
        self.objaverse_lvis_acc_top_3.reset()

        acc = self.objaverse_lvis_acc_top_5.compute()
        self.log(f"val_eval/objaverse_lvis_acc_top_5", acc, sync_dist=True)
        self.objaverse_lvis_acc_top_5.reset()
