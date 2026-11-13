# Sparse Window Attention

Reference implementation for **Sparse Window Attention for Long-Context
Transformers** (S. Gomez, K. Osei; NeurIPS 2024).

The released v1.0 implementation is available in `src/swa/`. Sliding-window
attention kernels and the training loop are included.

## Usage

```bash
pip install -r requirements.txt
python -m swa.train --config configs/base.yaml
```
