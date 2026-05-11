# SPDX-License-Identifier: MIT
"""MAE pretraining loop.

CLI:
    python src/train_pretrain.py --config config/pretrain.yaml --base . --data data

Python:
    from train_pretrain import main as pretrain_main
    pretrain_main(cfg, base_dir, data_dir)
"""

import argparse
import csv
import os
import time

import torch
import yaml

from data import load_and_split_pretrain, create_pretrain_dataloaders
from lr_schedule import WarmUpCosine
from model import build_mae_from_config
from utils import save_config


def main(cfg, base_dir, data_dir):
    """Run MAE pretraining."""
    model_cfg = cfg["model"]
    pre_cfg = cfg["pretrain"]
    paths = cfg["paths"]

    output_dir = os.path.join(base_dir, paths["output_subdir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pretrain_path = os.path.join(data_dir, paths["pretrain_file"])
    x_train, x_val, x_test = load_and_split_pretrain(
        pretrain_path,
        split=pre_cfg["split"],
        seed=pre_cfg["split_seed"],
    )

    preload_device = device if cfg["preload_gpu"] else None
    train_loader, val_loader, test_loader = create_pretrain_dataloaders(
        x_train, x_val, x_test,
        batch_size=pre_cfg["batch_size"],
        preload_device=preload_device,
    )

    build_cfg = {
        **model_cfg,
        "mask_proportion": pre_cfg["mask_proportion"],
        "dropout": pre_cfg["dropout"],
    }
    mae_model = build_mae_from_config(build_cfg, device=device)

    opt_cfg = pre_cfg["optimizer"]
    total_steps = len(train_loader) * pre_cfg["epochs"]
    warmup_steps = int(total_steps * pre_cfg["warmup_epoch_percentage"])

    print(f"total_steps: {total_steps}, warmup_steps: {warmup_steps}")

    lr_schedule = WarmUpCosine(
        learning_rate_base=opt_cfg["learning_rate"],
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    optimizer = torch.optim.AdamW(
        mae_model.parameters(),
        lr=opt_cfg["learning_rate"],
        weight_decay=opt_cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: lr_schedule(step) / opt_cfg["learning_rate"],
    )

    model_name = paths["model_name"]
    os.makedirs(output_dir, exist_ok=True)
    save_config(cfg, output_dir, model_name + "_config.yaml")

    log_path = os.path.join(output_dir, model_name + "_log.csv")
    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(["epoch", "loss", "mae", "val_loss", "val_mae", "lr"])

    for epoch in range(pre_cfg["epochs"]):
        epoch_start = time.time()

        mae_model.train()
        train_loss_sum = 0.0
        train_mae_sum = 0.0
        n_train_samples = 0
        n_train_mae_elems = 0

        for (signals_batch,) in train_loader:
            signals_batch = signals_batch.to(device)

            optimizer.zero_grad()
            loss, target_masked, pred_masked = mae_model(signals_batch)
            loss.backward()

            lr_last = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()

            bsz = signals_batch.size(0)
            train_loss_sum += loss.item() * bsz
            n_train_samples += bsz

            errors = torch.abs(pred_masked.detach() - target_masked.detach())
            train_mae_sum += errors.sum().item()
            n_train_mae_elems += errors.numel()

        train_loss_epoch = train_loss_sum / n_train_samples
        train_mae_epoch = train_mae_sum / n_train_mae_elems

        mae_model.eval()
        val_loss_sum = 0.0
        val_mae_sum = 0.0
        n_val_samples = 0
        n_val_mae_elems = 0

        with torch.no_grad():
            for (signals_batch,) in val_loader:
                signals_batch = signals_batch.to(device)
                loss, target_masked, pred_masked = mae_model.calculate_loss(signals_batch)

                bsz = signals_batch.size(0)
                val_loss_sum += loss.item() * bsz
                n_val_samples += bsz

                errors = torch.abs(pred_masked - target_masked)
                val_mae_sum += errors.sum().item()
                n_val_mae_elems += errors.numel()

        val_loss_epoch = val_loss_sum / n_val_samples
        val_mae_epoch = val_mae_sum / n_val_mae_elems

        epoch_sec = time.time() - epoch_start
        print(
            f"Epoch {epoch:03d} | "
            f"loss: {train_loss_epoch:.6f}, mae: {train_mae_epoch:.6f} | "
            f"val_loss: {val_loss_epoch:.6f}, val_mae: {val_mae_epoch:.6f} | "
            f"lr: {lr_last:.2e} | "
            f"{epoch_sec:.1f}s",
        )

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            writer.writerow([
                epoch,
                train_loss_epoch,
                train_mae_epoch,
                val_loss_epoch,
                val_mae_epoch,
                lr_last,
            ])

    mae_model.eval()
    test_loss_sum = 0.0
    test_mae_sum = 0.0
    n_test = 0
    n_elems = 0

    with torch.no_grad():
        for (signals_batch,) in test_loader:
            signals_batch = signals_batch.to(device)
            loss, target_masked, pred_masked = mae_model.calculate_loss(signals_batch)
            bsz = signals_batch.size(0)

            test_loss_sum += loss.item() * bsz
            n_test += bsz

            errors = torch.abs(pred_masked - target_masked)
            test_mae_sum += errors.sum().item()
            n_elems += errors.numel()

    test_loss = test_loss_sum / n_test
    test_mae = test_mae_sum / n_elems

    print(f"\nTest Loss: {test_loss:.5f}")
    print(f"Test MAE:  {test_mae:.5f}")

    save_path = os.path.join(output_dir, model_name + ".pth")
    torch.save(mae_model.state_dict(), save_path)
    print(f"{save_path} --> saved")


def cli():
    parser = argparse.ArgumentParser(description="MAE Pretraining")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--base", type=str, required=True,
                        help="Base directory for outputs")
    parser.add_argument("--data", type=str, required=True,
                        help="Directory containing the pretrain NPZ")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.base, args.data)


if __name__ == "__main__":
    cli()
