# ZFNet course project (with LeNet-5)

PyTorch implementations of **LeNet-5** and a **CIFAR-10–adapted ZFNet** for studying how early CNN architectures differ in inductive bias, capacity, and compute.

## Team Members

- Birhanu Asmamaw Baye — UGR/2204/13
- Sifan Fita — UGR/8856/14
- Sura Itana — UGR/2347/14

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

For GPU wheels, install PyTorch from the official selector instead of the CPU-only `requirements.txt` pins if needed.

## Train / evaluate

```bash
# example quick runs
.venv/bin/python train.py --model lenet --epochs 10 --data-dir ./data
.venv/bin/python train.py --model zfnet --epochs 10 --data-dir ./data

# longer (better) training with SGD + LR schedule and TensorBoard logs
.venv/bin/python train.py --model zfnet --epochs 50 --optimizer sgd --lr 0.01 --batch-size 128

# view logs with TensorBoard
# tensorboard --logdir checkpoints/runs
```

Checkpoints are written to `./checkpoints/`. CIFAR-10 is downloaded into `./data/` on first run.

Recent run results:

- LeNet-5 on MNIST reached a best test accuracy of **99.09%** after 10 epochs.

## Report

See `report.md` for background on LeNet-5, AlexNet as predecessor, ZFNet differences, strengths/weaknesses, and a table to paste your measured accuracies. A NotebookLM study link can be kept in that document for your group.

## Layout

- `models/lenet.py` — LeNet-5-style CNN with inline architectural comments.
- `models/zfnet.py` — ZFNet-inspired CNN with inline comments explaining AlexNet lineage.
- `train.py` — CIFAR-10 training loop, logging, best checkpoint saving.
