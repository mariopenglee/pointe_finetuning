import yaml
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from pointe_finetune.dataset import PointCloudTextDataset
from pointe_finetune.model_utils import load_pointe_model, load_pointe_diffusion


def main(config_path: str):
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg['device'] if torch.cuda.is_available() else 'cpu')

    # Dataset & DataLoader
    ds_cfg = cfg['data']
    dataset = PointCloudTextDataset(
        csv_path=ds_cfg['csv_path'],
        pc_dir=ds_cfg['pc_dir'],
        preload=ds_cfg['preload'],
        device=device
    )
    loader = DataLoader(
        dataset,
        batch_size=ds_cfg['batch_size'],
        shuffle=ds_cfg['shuffle'],
        num_workers=ds_cfg['num_workers'],
        pin_memory=True,
    )

    # Models & Diffusions
    base_model = load_pointe_model(cfg['model']['base_model'], device).train()
    diffusion = load_pointe_diffusion(cfg['model']['base_model'])

    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=cfg['training']['weight_decay']
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=cfg['training']['scheduler']['factor'],
        patience=cfg['training']['scheduler']['patience'],
        verbose=True
    )

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, cfg['training']['epochs'] + 1):
        epoch_loss = 0.0
        for step, (prompts, x_start) in enumerate(loader, 1):
            x_start = x_start.to(device)
            t = torch.randint(
                0,
                diffusion.num_timesteps,
                (x_start.size(0),),
                device=device,
            )
            loss_dict = diffusion.training_losses(
                base_model,
                x_start,
                t,
                model_kwargs={"texts": prompts}
            )
            loss = loss_dict['loss'].mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % 10 == 0:
                print(f"[Epoch {epoch:02d}] Step {step}/{len(loader)} Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(loader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch} avg loss: {avg_loss:.4f}")

        # Early stopping
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg['training']['early_stop_patience']:
                print(f"No improvement for {cfg['training']['early_stop_patience']} epochs, stopping.")
                break

    torch.save(base_model.state_dict(), 'pointe_finetuned.pt')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="config/config.yaml")
    args = p.parse_args()
    main(args.config)