# SPDX-License-Identifier: MIT
"""Downstream training (linear probing or full fine-tuning) with a pretrained MAE encoder.

downstream.task_type selects regression (MSE, R², MAE) or classification (CE, top-k).
"""

import argparse
import csv
import os
import time

import torch
import torch.nn as nn
import yaml

from data import load_and_split_probing, create_downstream_dataloaders
from lr_schedule import WarmUpCosine
from model import build_mae_from_config, build_downstream_from_mae
from utils import save_config


def compute_r_squared(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot == 0:
        return 0.0
    return (1.0 - ss_res / ss_tot).item()


def compute_mae_metric(y_true, y_pred):
    return (y_true - y_pred).abs().mean().item()


def topk_correct(logits, targets, ks=(1,)):
    """Top-k correct counts. Returns {k: correct_count}."""
    maxk = max(ks)
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))
    return {k: correct[:k].reshape(-1).float().sum().item() for k in ks}


def _build_param_groups(model, base_lr, llrd_factor, weight_decay):
    """Optimizer param groups with layer-wise LR decay.

    Scaling (head -> deepest):
        head, encoder.final_norm:  base_lr
        encoder.layers[-1]:        base_lr * factor^1
        ...
        encoder.layers[0]:         base_lr * factor^num_layers
        patch_encoder:             base_lr * factor^(num_layers+1)

    llrd_factor=1.0 disables LLRD (all groups share base_lr).
    """
    groups = []
    num_layers = len(model.encoder.layers)

    groups.append({
        "params": [p for p in model.head.parameters() if p.requires_grad],
        "lr": base_lr, "lr_scale": 1.0, "weight_decay": weight_decay,
        "name": "head",
    })
    groups.append({
        "params": [p for p in model.encoder.final_norm.parameters() if p.requires_grad],
        "lr": base_lr, "lr_scale": 1.0, "weight_decay": weight_decay,
        "name": "encoder.final_norm",
    })

    for i in range(num_layers):
        layer_idx = num_layers - 1 - i
        scale = llrd_factor ** (i + 1)
        params = [p for p in model.encoder.layers[layer_idx].parameters()
                  if p.requires_grad]
        if params:
            groups.append({
                "params": params,
                "lr": base_lr * scale, "lr_scale": scale,
                "weight_decay": weight_decay,
                "name": f"encoder.layer{layer_idx}",
            })

    scale = llrd_factor ** (num_layers + 1)
    params = [p for p in model.patch_encoder.parameters() if p.requires_grad]
    if params:
        groups.append({
            "params": params,
            "lr": base_lr * scale, "lr_scale": scale,
            "weight_decay": weight_decay,
            "name": "patch_encoder",
        })

    return [g for g in groups if g["params"]]


def main(cfg, base_dir, data_dir, factor_name=None):
    """Run downstream training on a pretrained MAE encoder."""
    ds_cfg = cfg["downstream"]
    paths = cfg["paths"]

    if factor_name is None:
        factor_name = ds_cfg["factor"]

    task_type = ds_cfg["task_type"]
    if task_type not in ("regression", "classification"):
        raise ValueError(
            f"downstream.task_type must be 'regression' or 'classification', "
            f"got '{task_type}'."
        )
    output_dim = ds_cfg["num_classes"]

    output_dir = os.path.join(base_dir, paths["output_subdir"])
    pretrain_dir = os.path.join(base_dir, paths["pretrain_subdir"])
    pretrained_weights = os.path.join(pretrain_dir, paths["pretrain_name"] + ".pth")
    pretrained_cfg_path = os.path.join(pretrain_dir, paths["pretrain_name"] + "_config.yaml")

    # Architecture comes from the pretrain config saved next to the .pth weights.
    with open(pretrained_cfg_path, "r") as f:
        model_cfg = yaml.safe_load(f)["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task type: {task_type}, output_dim: {output_dim}")
    print(f"Factor: {factor_name}")

    ds_file = os.path.join(data_dir, paths["downstream_file"])
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_split_probing(
        ds_file, factor_name,
        split=ds_cfg["split"], seed=ds_cfg["split_seed"],
    )

    preload_device = device if cfg["preload_gpu"] else None
    train_loader, val_loader, test_loader = create_downstream_dataloaders(
        x_train, y_train, x_val, y_val, x_test, y_test,
        batch_size=ds_cfg["batch_size"],
        preload_device=preload_device,
        task_type=task_type,
    )

    # mask_proportion and dropout are only used at forward time and have no
    # effect on the loaded state_dict shape; downstream overrides them anyway.
    build_cfg = {**model_cfg, "mask_proportion": 0.0, "dropout": 0.0}
    mae_model = build_mae_from_config(build_cfg, device=device)

    model_name = paths["model_name"]
    state_dict = torch.load(pretrained_weights, map_location=device, weights_only=True)
    mae_model.load_state_dict(state_dict)
    print(f"Pretrained weights loaded from: {pretrained_weights}")

    downstream_model = build_downstream_from_mae(
        mae_model, output_dim, model_cfg, device=device,
        finetune_strategy=ds_cfg["finetune_strategy"],
        downstream_dropout=ds_cfg["dropout"],
        head_dropout=ds_cfg["head_dropout"],
        train_patch_encoder=ds_cfg["train_patch_encoder"],
    )

    criterion = nn.MSELoss() if task_type == "regression" else nn.CrossEntropyLoss()

    opt_cfg = ds_cfg["optimizer"]
    base_lr = opt_cfg["learning_rate"]
    llrd_factor = ds_cfg["llrd_factor"]

    param_groups = _build_param_groups(downstream_model, base_lr, llrd_factor,
                                       opt_cfg["weight_decay"])

    optimizer = torch.optim.SGD(
        param_groups,
        momentum=opt_cfg["momentum"],
        nesterov=opt_cfg["nesterov"],
    )

    if llrd_factor != 1.0:
        print(f"LLRD enabled (factor={llrd_factor}):")
        for g in optimizer.param_groups:
            print(f"  {g['name']:20s}  lr_scale={g['lr_scale']:.4f}  lr={g['lr']:.6f}")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * ds_cfg["epochs"]
    warmup_steps = int(total_steps * ds_cfg["warmup_epoch_percentage"])

    lr_schedule = WarmUpCosine(
        learning_rate_base=base_lr,
        total_steps=total_steps,
        warmup_learning_rate=0.0,
        warmup_steps=warmup_steps,
    )

    print(f"Steps/epoch: {steps_per_epoch}, total: {total_steps}, warmup: {warmup_steps}")

    global_step = 0

    os.makedirs(output_dir, exist_ok=True)
    save_config(cfg, output_dir, model_name + "_config.yaml")

    log_path = os.path.join(output_dir, model_name + "_log.csv")

    if task_type == "regression":
        header = ["epoch", "loss", "r2", "mae_metric", "val_loss", "val_r2", "val_mae_metric"]
    else:
        ks = (1, 2, 3, 4, 5) if output_dim > 5 else tuple(range(1, min(output_dim, 5) + 1))
        header = (["epoch", "loss"]
                  + [f"top_{k}" for k in ks]
                  + ["val_loss"]
                  + [f"val_top_{k}" for k in ks])

    with open(log_path, mode="w", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        writer.writerow(header)

    for epoch in range(ds_cfg["epochs"]):
        epoch_start = time.time()

        downstream_model.train()
        train_loss_sum = 0.0
        n_train = 0

        if task_type == "regression":
            all_preds_train, all_targets_train = [], []
        else:
            train_correct = {k: 0.0 for k in ks}

        for signals_batch, labels_batch in train_loader:
            signals_batch = signals_batch.to(device)
            labels_batch = labels_batch.to(device)

            lr = lr_schedule(global_step)
            for g in optimizer.param_groups:
                g["lr"] = lr * g["lr_scale"]
            global_step += 1

            optimizer.zero_grad()
            output = downstream_model(signals_batch)

            if task_type == "regression":
                pred = output.squeeze(-1)
                loss = criterion(pred, labels_batch)
                all_preds_train.append(pred.detach())
                all_targets_train.append(labels_batch.detach())
            else:
                loss = criterion(output, labels_batch)
                correct_dict = topk_correct(output, labels_batch, ks=ks)
                for k in ks:
                    train_correct[k] += correct_dict[k]

            loss.backward()
            optimizer.step()

            bsz = labels_batch.size(0)
            train_loss_sum += loss.item() * bsz
            n_train += bsz

        train_loss_epoch = train_loss_sum / n_train

        if task_type == "regression":
            all_p = torch.cat(all_preds_train)
            all_t = torch.cat(all_targets_train)
            train_r2 = compute_r_squared(all_t, all_p)
            train_mae = compute_mae_metric(all_t, all_p)
        else:
            train_top = {k: train_correct[k] / n_train for k in ks}

        downstream_model.eval()
        val_loss_sum = 0.0
        n_val = 0

        if task_type == "regression":
            all_preds_val, all_targets_val = [], []
        else:
            val_correct = {k: 0.0 for k in ks}

        with torch.no_grad():
            for signals_batch, labels_batch in val_loader:
                signals_batch = signals_batch.to(device)
                labels_batch = labels_batch.to(device)

                output = downstream_model(signals_batch)

                if task_type == "regression":
                    pred = output.squeeze(-1)
                    loss = criterion(pred, labels_batch)
                    all_preds_val.append(pred)
                    all_targets_val.append(labels_batch)
                else:
                    loss = criterion(output, labels_batch)
                    correct_dict = topk_correct(output, labels_batch, ks=ks)
                    for k in ks:
                        val_correct[k] += correct_dict[k]

                bsz = labels_batch.size(0)
                val_loss_sum += loss.item() * bsz
                n_val += bsz

        val_loss_epoch = val_loss_sum / n_val

        if task_type == "regression":
            all_p = torch.cat(all_preds_val)
            all_t = torch.cat(all_targets_val)
            val_r2 = compute_r_squared(all_t, all_p)
            val_mae = compute_mae_metric(all_t, all_p)
        else:
            val_top = {k: val_correct[k] / n_val for k in ks}

        epoch_sec = time.time() - epoch_start

        if task_type == "regression":
            print(
                f"Epoch {epoch:03d} | "
                f"loss: {train_loss_epoch:.6f}, R²: {train_r2:.4f}, MAE: {train_mae:.4f} | "
                f"val_loss: {val_loss_epoch:.6f}, val_R²: {val_r2:.4f}, val_MAE: {val_mae:.4f} | "
                f"{epoch_sec:.1f}s",
            )
        else:
            print(
                f"Epoch {epoch:03d} | "
                f"loss: {train_loss_epoch:.4f}, top1: {train_top[1]:.4f} | "
                f"val_loss: {val_loss_epoch:.4f}, val_top1: {val_top[1]:.4f} | "
                f"{epoch_sec:.1f}s",
            )

        with open(log_path, mode="a", newline="") as f:
            writer = csv.writer(f, delimiter=",")
            if task_type == "regression":
                writer.writerow([epoch, train_loss_epoch, train_r2, train_mae,
                                 val_loss_epoch, val_r2, val_mae])
            else:
                writer.writerow(
                    [epoch, train_loss_epoch]
                    + [train_top[k] for k in ks]
                    + [val_loss_epoch]
                    + [val_top[k] for k in ks]
                )

    downstream_model.eval()
    test_loss_sum = 0.0
    n_test = 0

    if task_type == "regression":
        all_preds_test, all_targets_test = [], []
    else:
        test_correct = {k: 0.0 for k in ks}

    with torch.no_grad():
        for signals_batch, labels_batch in test_loader:
            signals_batch = signals_batch.to(device)
            labels_batch = labels_batch.to(device)

            output = downstream_model(signals_batch)

            if task_type == "regression":
                pred = output.squeeze(-1)
                loss = criterion(pred, labels_batch)
                all_preds_test.append(pred)
                all_targets_test.append(labels_batch)
            else:
                loss = criterion(output, labels_batch)
                correct_dict = topk_correct(output, labels_batch, ks=ks)
                for k in ks:
                    test_correct[k] += correct_dict[k]

            bsz = labels_batch.size(0)
            test_loss_sum += loss.item() * bsz
            n_test += bsz

    test_loss = test_loss_sum / n_test

    print(f"\nTest results [{factor_name}]:")
    print(f"  loss: {test_loss:.6f}")
    if task_type == "regression":
        all_p = torch.cat(all_preds_test)
        all_t = torch.cat(all_targets_test)
        test_r2 = compute_r_squared(all_t, all_p)
        test_mae = compute_mae_metric(all_t, all_p)
        print(f"  R²:   {test_r2:.4f}")
        print(f"  MAE:  {test_mae:.4f}")
    else:
        test_top = {k: test_correct[k] / n_test for k in ks}
        for k in ks:
            print(f"  top{k}: {test_top[k]:.4f}")

    save_path = os.path.join(output_dir, model_name + ".pth")
    torch.save(downstream_model.state_dict(), save_path)
    print(f"{save_path} --> saved")


def cli():
    parser = argparse.ArgumentParser(description="MAE downstream training")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--base", type=str, required=True)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--factor", type=str, default=None,
                        help="Label key in the probing NPZ (overrides config)")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    main(cfg, args.base, args.data, factor_name=args.factor)


if __name__ == "__main__":
    cli()
