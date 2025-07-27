import yaml
import torch
from point_e.diffusion.sampler import PointCloudSampler
from pointe_finetune.model_utils import load_pointe_model, load_pointe_diffusion
from pointe_finetune.dataset import PointCloudTextDataset


def main(config_path: str, prompt: str, output_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    # Load models
    base_model = load_pointe_model(cfg['model']['base_model'], device).eval()
    up_model   = load_pointe_model(cfg['model']['upsample_model'], device).eval()

    base_diff = load_pointe_diffusion(cfg['model']['base_model'])
    up_diff   = load_pointe_diffusion(cfg['model']['upsample_model'])

    sampler = PointCloudSampler(
        device=device,
        models=[base_model, up_model],
        diffusions=[base_diff, up_diff],
        num_points=[1024, 4096-1024],
        aux_channels=['R','G','B'],
        guidance_scale=cfg['model']['guidance_scale'],
        model_kwargs_key_filter=('texts','')
    )

    samples = None
    for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs={'texts':[prompt]}):
        samples = x

    pcs = sampler.output_to_point_clouds(samples)
    torch.save(pcs[0].to_tensor(), output_path)
    print(f"Saved point cloud to {output_path}")


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="output_pc.pt")
    args = p.parse_args()
    main(args.config, args.prompt, args.out)