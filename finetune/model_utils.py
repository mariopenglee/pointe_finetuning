import torch
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config


def load_pointe_model(model_name: str, device: torch.device):
    model = model_from_config(MODEL_CONFIGS[model_name], device)
    model.load_state_dict(load_checkpoint(model_name, device))
    return model


def load_pointe_diffusion(config_name: str):
    return diffusion_from_config(DIFFUSION_CONFIGS[config_name])