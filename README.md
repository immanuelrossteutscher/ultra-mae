# ultra-mae

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/immanuelrossteutscher/ultra-mae/blob/main/notebooks/quickstart.ipynb)

A small, self-contained 1D Masked Autoencoder for ultrasound burst signals,
intended as a starting point for anyone working with similar 1D signal data.

> **Reference paper:** _An Ultrasound Masked Autoencoder for Universal Feature
> Extraction from Burst Signals_, IEEE Access, December 2025.

## What's in here

```
ultra-mae/
├── src/                model + training scripts (pure PyTorch)
├── config/             pretrain.yaml + linear_probe.yaml
├── data/               pretrain.npz (30k) + probing.npz (3k, position label)
├── notebooks/          quickstart end-to-end
├── requirements.txt
├── LICENSE             MIT
└── README.md
```

The included NPZ files are synthetic ultrasound burst signals with five
varied factors (frequency, amplitude, SNR, burst length, position). The
pretrain file holds **signals only**; the probing file holds signals plus
the **position** label, used as a regression target.

## Architecture (~1.3 M parameters)

```
Signal (1×512)
   │  Patches: 16 patches × 32 samples
   ▼
PatchEncoder         linear projection + positional embedding + masking (75% pretraining)
   ▼
MAEEncoder           6 Transformer blocks (dim 128, 4 heads, head_dim 64, mlp_ratio 2)
   ▼
   ├── (pretraining) MAEDecoder: 2 Transformer blocks (dim 64) → reconstruct masked patches
   └── (downstream)  LayerNorm → Global Average Pool → Head (linear or full FT)
```

Defaults are in `config/pretrain.yaml`. To adapt to a different signal length,
change `model.signal_size` and `model.patch_size` (signal_size must be
divisible by patch_size).

## Quickstart

```bash
pip install -r requirements.txt

# 1) Pretrain the MAE on the included pretrain.npz
python src/train_pretrain.py --config config/pretrain.yaml --base . --data data

# 2) Linear-probe the encoder on the position label
python src/train_downstream.py --config config/linear_probe.yaml --base . --data data --factor position
```

Outputs land in `logs/pretrain/` and `logs/probing/`:

- `mae_pre.pth` — pretrained MAE weights
- `mae_position_lp.pth` — fine-tuned probing model
- `*_log.csv` — per-epoch metrics
- `*_config.yaml` — exact config used (for reproducibility)

For an end-to-end walk-through with plots and reconstruction visualization,
run `notebooks/quickstart.ipynb` on Colab via the badge above — the first
notebook cell clones the repo automatically.

## Using your own data

The training scripts expect NPZ files with this minimal schema:

| File           | Required keys                                        |
| -------------- | ---------------------------------------------------- |
| pretrain NPZ   | `signals`: float32 array of shape `(n, signal_size)` |
| probing NPZ    | `signals` + one or more label arrays of shape `(n,)` |

Signals should be roughly normalized to `[0, 1]`. To use a different label,
set `downstream.factor` in `config/linear_probe.yaml` (or pass `--factor
<name>` on the CLI). The label name must match a key in your probing NPZ.

To switch from regression to classification, set in
`config/linear_probe.yaml`:

```yaml
downstream:
  task_type: classification
  num_classes: K          # number of classes
```

The label array must then contain integer class IDs. The bundled
`probing.npz` only ships regression targets, so you will need your own
probing NPZ — for example:

```python
import numpy as np
np.savez(
    "my_probing.npz",
    signals=signals,                          # (n, signal_size) float32
    my_class=np.asarray(class_ids, np.int64), # (n,) integer class IDs
)
```

Then point `paths.downstream_file` at `my_probing.npz` and set
`downstream.factor: my_class`.

To switch from linear probing to full fine-tuning:

```yaml
downstream:
  finetune_strategy: full
  llrd_factor: 0.75       # optional ViT-style layer-wise LR decay
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@article{rossteutscher2025ultrasoundmae,
  title   = {An Ultrasound Masked Autoencoder for Universal Feature Extraction from Burst Signals},
  author  = {Rossteutscher, Immanuel},
  journal = {IEEE Access},
  year    = {2025},
  month   = {December}
}
```
