import argparse

import torch
import yaml
from torch.optim import AdamW

from .model import SparseWindowTransformer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model_cfg = config["model"]
    model = SparseWindowTransformer(
        vocab_size=32000,
        hidden_size=model_cfg["hidden_size"],
        num_heads=model_cfg["num_heads"],
        num_layers=model_cfg["num_layers"],
        window_size=model_cfg["window_size"],
    )
    optimizer = AdamW(model.parameters(), lr=config["train"]["learning_rate"])
    print(model)
    print("optimizer ready:", type(optimizer).__name__)


if __name__ == "__main__":
    main()
