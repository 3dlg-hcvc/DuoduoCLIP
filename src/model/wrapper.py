from huggingface_hub import hf_hub_download

from src.model.duoduoclip import DuoduoCLIP

def get_model(filename, device='cuda'):
    ckpt_path = hf_hub_download(repo_id='3dlg-hcvc/DuoduoCLIP', filename=filename)
    duoduoclip = DuoduoCLIP.load_from_checkpoint(ckpt_path)
    duoduoclip.eval()
    duoduoclip.to(device)
    return duoduoclip
