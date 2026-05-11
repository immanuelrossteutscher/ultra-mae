# SPDX-License-Identifier: MIT
"""Shared utilities."""

import os

import yaml


def save_config(cfg, output_dir, filename):
    """Save a config dict as YAML next to the model weights."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Config saved to: {path}")
