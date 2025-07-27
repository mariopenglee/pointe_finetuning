import yaml
import torch
from point_e.diffusion.sampler import PointCloudSampler
from finetune.model_utils import load_pointe_model, load_pointe_diffusion
from finetune.dataset import PointCloudTextDataset
from tqdm import tqdm
import subprocess
import os
from point_e.util.point_cloud import PointCloud


def main(config_path: str, prompt: str, output_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    # Load models
    print("Loading models...")
    base_model = load_pointe_model(cfg['model']['base_model'], device).eval()
    # Load fine-tuned weights if available
    finetuned_ckpt = 'pointe_finetuned.pt'
    if os.path.exists(finetuned_ckpt):
        print(f"Loading fine-tuned weights from {finetuned_ckpt}")
        base_model.load_state_dict(torch.load(finetuned_ckpt, map_location=device))
    up_model   = load_pointe_model(cfg['model']['upsample_model'], device).eval()

    base_diff = load_pointe_diffusion(cfg['model']['base_model'])
    up_diff   = load_pointe_diffusion(cfg['model']['upsample_model'])

    print("Creating sampler...")
    sampler = PointCloudSampler(
        device=device,
        models=[base_model, up_model],
        diffusions=[base_diff, up_diff],
        num_points=[1024, 4096-1024],
        aux_channels=['R','G','B'],
        guidance_scale=cfg['model']['guidance_scale'],
        model_kwargs_key_filter=('texts','')
    )

    print("Sampling point cloud...")
    samples = None
    # Use tqdm to show progress in the sampling loop
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs={'texts':[prompt]}), desc="Sampling"):
        samples = x

    pcs = sampler.output_to_point_clouds(samples)
    # Save using PointCloud.save (npz format)
    pc_obj = pcs[0]
    out_ext = os.path.splitext(output_path)[1]
    if out_ext == '.npz':
        pc_obj.save(output_path)
    else:
        # fallback to torch for .pt, but recommend .npz
        import warnings
        warnings.warn('Saving as .pt is deprecated, use .npz for PointClouds!')
        torch.save(pc_obj, output_path)
    print(f"Saved point cloud to {output_path}")

    # Visualize the saved point cloud
    subprocess.run([
        'python',
        os.path.join(os.path.dirname(__file__), '../scripts/visualize_pointclouds.py'),
        '--dir', os.path.dirname(output_path),
        '--file', os.path.basename(output_path)
    ])


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--out", type=str, default="outputs/output.npz")
    args = p.parse_args()
    main(args.config, args.prompt, args.out)