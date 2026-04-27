#!/usr/bin/env python3
"""Entry point generico per lanciare un esperimento da config YAML."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path al config YAML dell'esperimento")
    parser.add_argument("--seed", type=int, default=None,
                        help="Override del seed (per multi-run)")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments"),
                        help="Root delle directory di output")
    args = parser.parse_args()

    # TODO:
    # 1. carica config (con composizione hydra-style)
    # 2. crea output_dir/<exp_name>_seed<seed>/
    # 3. salva snapshot della config in quella directory
    # 4. setup logging + mlflow
    # 5. costruisci dati, modello, strategia, server FL
    # 6. esegui training
    # 7. salva metriche e checkpoint
    raise NotImplementedError


if __name__ == "__main__":
    main()
