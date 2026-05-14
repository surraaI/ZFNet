# ZFNet course project (with LeNet-5)

PyTorch implementations of **LeNet-5** and a **CIFAR-10–adapted ZFNet** for studying how early CNN architectures differ in inductive bias, capacity, and compute.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For GPU wheels, install PyTorch from the official selector instead of the CPU-only `requirements.txt` pins if needed.

## Train / evaluate

```bash
.venv/bin/python train.py --model lenet --epochs 10 --data-dir ./data
.venv/bin/python train.py --model zfnet --epochs 10 --data-dir ./data
```

Checkpoints are written to `./checkpoints/`. CIFAR-10 is downloaded into `./data/` on first run.

## Report

See `report.md` for background on LeNet-5, AlexNet as predecessor, ZFNet differences, strengths/weaknesses, and a table to paste your measured accuracies. A NotebookLM study link can be kept in that document for your group.

## Layout

- `models/lenet.py` — LeNet-5-style CNN with inline architectural comments.
- `models/zfnet.py` — ZFNet-inspired CNN with inline comments explaining AlexNet lineage.
- `train.py` — CIFAR-10 training loop, logging, best checkpoint saving.
