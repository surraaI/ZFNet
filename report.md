# CNN Evolution: LeNet-5, AlexNet, and ZFNet

This report supports the course assignment on **ZFNet** with **LeNet-5** as the historical baseline. It traces how convolutional networks moved from digit recognition to large-scale classification, how **AlexNet** set the stage for **ZFNet**, and how visualization changed design practice. The narrative aligns with the group’s NotebookLM synthesis, with explicit notes on **how this repository’s code matches or simplifies** the original papers.

---

## I. Introduction to Convolutional Neural Networks (CNNs)

### The paradigm shift

Before widespread CNN adoption, many vision pipelines relied on **hand-crafted features** (edges, SIFT/HOG-like descriptors, bag-of-words) followed by a separate classifier. CNNs reversed the division of labor: **convolutional layers learn feature detectors from data**, and deeper layers compose them into increasingly global patterns. Shared weights make this parameter-efficient compared with treating every pixel independently in a fully connected network.

### Assignment context

This project traces that evolution in three steps:

1. **LeNet-5 (1998)** — established the classic pattern for small images (e.g., 32×32 digits).
2. **AlexNet (2012)** — scaled depth, data, and compute to ImageNet-scale RGB classification.
3. **ZFNet (2013)** — used **visualization** to diagnose AlexNet and refine its architecture (especially layer 1), moving design from pure trial-and-error toward **evidence-based** changes.

---

## II. LeNet-5: The Pioneer (1998)

### Motivation

LeNet-5 (LeCun *et al.*) was developed for **document and check recognition**, especially **handwritten digits** (the MNIST-style setting in the original work). The goal was to replace fragile pipelines of engineered features with a **single trainable system** that maps pixels to class decisions.

### Architectural innovations

- **Layer sequence:** The pattern **Input → Convolution (C) → Subsampling / Pooling (S) → … → Full connection (F)** became the template for later CNNs.
- **Local receptive fields:** Small spatial kernels (classically **5×5**) detect local structure (strokes, corners) everywhere on the grid.
- **Weight sharing:** The same filter bank slides over the image, cutting parameters versus fully connected layers and encoding **translation equivariance** in early layers.
- **Partial connectivity (layer C3 in the original paper):** Some feature maps in C3 connect only to selected maps in S2 to reduce computation and encourage diversity of features. Many modern reproductions use **dense** convolutions for simplicity; the *idea* (structured connectivity for efficiency) remains historically important.
- **Output layer in the original:** The paper describes **Euclidean RBF** units tied to prototype digit masks. Modern practice almost always uses a **linear + softmax / cross-entropy** classifier instead.

### Historical impact and tradeoffs

LeNet showed that CNNs could reach **very low error on digits** with modest compute by today’s standards. Limitations of the era included **dataset size**, **hardware**, and shallower networks than modern ImageNet winners. Original activations were often **tanh/sigmoid** and pooling was closer to **averaging/subsampling**; later work favored **ReLU** and **max pooling** for speed and optimization behavior.

### Implementation in this repository (`models/lenet.py`)

| Paper / NotebookLM idea | What we implemented | Why |
| --- | --- | --- |
| 32×32 inputs, digit-scale | **32×32 RGB** CIFAR-10 (`in_channels=3`) | Same spatial size as CIFAR; assignment asks for a realistic RGB benchmark. |
| C / S alternation | **Conv → max pool → conv → max pool → conv → FC → FC** | Preserves the pedagogical **hierarchy** while using max pool (stable, common in PyTorch coursework). |
| Tanh activations | **Tanh** after conv/FC blocks | Keeps a closer historical flavor than ReLU-only LeNet clones. |
| C3 partial connectivity | **Dense** `Conv2d(6, 16, …)` | Simpler code; comment in source notes the original sparse table. |

Annotated code lives in `models/lenet.py` and is trained via `train.py --model lenet`.

---

## III. The Bridge: AlexNet (ILSVRC 2012)

### Why AlexNet matters for ZFNet

**ZFNet is a direct architectural descendant of AlexNet.** Zeiler and Fergus started from the AlexNet blueprint and changed it where visualization showed **information loss** or **artifacts**—especially in early layers.

### Scaling up from LeNet

AlexNet applied LeNet’s ingredients at **ImageNet** scale (1.2M training images, 1000 classes, **224×224 RGB**):

- **Much greater depth and width** (millions of parameters).
- **ReLU** for faster training than saturating nonlinearities.
- **Dropout** (commonly **0.5** in large FC layers) to curb overfitting.
- **Heavy data augmentation** and **multi-GPU** training in the original implementation.

### Key differences from LeNet (summary)

| Aspect | LeNet-era practice | AlexNet |
| --- | --- | --- |
| Data | Small grayscale digits | Large natural RGB images |
| Nonlinearity | Tanh / sigmoid common | **ReLU** |
| Regularization | Architecture + dataset size | **Dropout**, augmentation |
| Compute | CPU-era feasible | **GPU** training at scale |

AlexNet’s **first convolution** used **11×11 filters** with **stride 4**—efficient for downsampling large inputs, but (as ZFNet argues) **risky for preserving mid-level spatial frequency content** in the first feature maps.

---

## IV. ZFNet: Visualizing and Understanding (2013)

### Motivation: the “black box” problem

AlexNet won ILSVRC 2012, yet many teams treated deep CNNs as **opaque stacks** tuned by guesswork. Zeiler and Fergus asked: **which patterns drive each layer**, and can we **edit the architecture** based on evidence rather than only on validation loss?

### Key innovation: Deconvolutional networks (Deconvnet)

A **Deconvnet** (attached as a “probe” to a trained CNN) maps **selected activations** back toward **input pixel space** so humans can see what image structure caused a neuron to fire.

**Forward pass bookkeeping:** During **max pooling**, the network records **switch variables**—which spatial position was the max in each pooling window.

**Three reconstruction steps (iterated up the hierarchy):**

1. **Unpooling:** Non-invertible pooling is approximated by scattering each upper-layer value back to the **winning location** recorded by the switches, restoring rough spatial layout.
2. **Rectification:** Apply **ReLU** so reconstructed maps stay **non-negative**, mirroring the forward pass.
3. **Filtering (transposed convolution view):** Use **transposed / flipped** versions of the learned filters to propagate activity “backward” through the layer, approximating an inverse of the forward convolution.

**What Deconvnet revealed (NotebookLM / paper themes):**

- **Hierarchy:** early layers → edges and color blobs; mid layers → textures; deep layers → class- and object-level parts with pose variability.
- **Training dynamics:** low-level filters stabilize quickly; class-specific structure in deep layers may need **many epochs**.
- **Diagnosis of AlexNet:** first-layer filters behaved as if they **skipped mid-frequency** structure; **large stride** in layer 1 correlated with **aliasing-like artifacts** in layer 2 feature visualizations—motivating ZFNet’s layer‑1 geometry changes.

### Architectural refinements vs AlexNet

**First layer (central ZFNet story):**

| Quantity | AlexNet (typical description) | ZFNet adjustment | Rationale from visualization |
| --- | --- | --- | --- |
| Filter size | **11×11** | **7×7** | Smaller fields capture **finer** local structure; 11×11 was “skipping” useful mid-frequency cues. |
| Stride | **4** | **2** | Less aggressive downsampling raises **effective sampling rate**, reducing **aliasing** and jagged / repetitive artifacts seen deeper when stride was too large. |

**Connectivity:** AlexNet’s original implementation split some layers across **two GPUs**, creating **sparse cross-map connectivity** in parts of the network. ZFNet’s published variant emphasizes **dense** connectivity through the conv trunk (layers 3–5), which matches the “single coherent feature hierarchy” story (implementation details differ by exact release, but the teaching point is: **remove artificial connectivity constraints** when they no longer serve hardware layout).

**First-layer filter renormalization (training detail):** The authors noted a few filters could **dominate** magnitudes. They **renormalized** convolution filters to a fixed **RMS radius** (reported as **0.1** in the narrative you summarized) when activations grew too large—especially important when inputs were scaled to a range like **[-128, 128]**. This is **not** reproduced in our minimal training script (we use standard CIFAR normalization instead).

### Strengths and weaknesses (ImageNet-era framing)

- **Strengths:** Strong **interpretability methodology** tied to measurable architecture edits; competitive **ILSVRC 2013** results; evidence that **ImageNet features transfer** to other datasets (a line of work that continued with later architectures).
- **Weaknesses:** Very large **fully connected** blocks (e.g., **4096** units) dominate **parameter count** and memory; training remains **data- and GPU-intensive** compared with LeNet-scale models.

### Computational tradeoffs (useful for slides)

A common teaching point (from your NotebookLM notes): in very large AlexNet/ZFNet-style models, **conv layers** can account for the bulk of **compute (FLOPs)** during forward/backward passes, while **FC layers** can still hold a **large fraction of parameters**—so “most parameters” and “most compute” are not always the same layers.

### Implementation in this repository (`models/zfnet.py`)

We implement a **CIFAR-10–adapted** ZFNet **inspired** by the paper’s **channel widths** (96 → 256 → 384 → 384 → 256), **ReLU**, **max pooling**, and **dropout 0.5** before the classifier head—matching the spirit of the NotebookLM “use ReLU + dropout 0.5” guidance.

**Intentional simplifications for a student codebase:**

| ImageNet ZFNet / NotebookLM | This repo | Reason |
| --- | --- | --- |
| **7×7**, stride **2** first conv on 224×224 | **5×5**, stride **1**, padding tuned for **32×32** | Keeps tensors well-defined on CIFAR while preserving “**smaller early kernels than AlexNet’s 11×11**” as a design lesson. |
| Large **FC** stacks (**4096** × 2) | **Global average pooling + one linear** | Cuts parameters so training fits laptops; still demonstrates the **conv trunk** pattern. |
| Deconvnet switches (`return_indices=True`) | **Not implemented** | Deconvnet is a **visualization tool**, not required for classification training; the report explains it, the code focuses on **train/eval**. |

Annotated code: `models/zfnet.py`. Training: `train.py --model zfnet`.

---

## V. Comparative Analysis and Evolution

### Task complexity

- **LeNet-5:** **32×32** (often **grayscale** digits), **10** classes, relatively **low intra-class variability** compared with natural photos.
- **ZFNet / AlexNet:** **224×224 RGB**, **1000** ImageNet classes, **large appearance variability**, requiring **capacity** and **regularization**.

### Design philosophy

- **LeNet era:** careful hand design of connectivity and subsampling for **efficiency** on constrained hardware.
- **AlexNet era:** scale depth, data, and GPUs; accept **opaque** behavior if scores improve.
- **ZFNet:** **measure internal representations** (Deconvnet, occlusion tests in the paper) and **refine architecture** where visualization shows **information bottlenecks** or **artifacts**.

### Occlusion (brief, from your NotebookLM)

Beyond Deconvnet, the authors used **occlusion** of input regions to show that **class probability and deep activations** track **true object parts** (e.g., dropping when an animal’s face is covered), supporting the claim that the network is not only exploiting background context.

---

## VI. Evaluation and Reproducible Experiments (This Project)

### Dataset

We train on **CIFAR-10** (50k train / 10k test, **32×32 RGB**, **10** classes) via `torchvision`. It is a standard **realistic** small-image benchmark for coursework—not a full ImageNet reproduction.

**Note on “corrupted” data files:** `data/cifar-10-batches-py/data_batch_*` are **binary pickle** archives. Opening them as text in an editor shows gibberish; that is **normal**, not corruption. If `pickle.load` ever fails, delete `data/cifar-10-batches-py` and `cifar-10-python.tar.gz` and rerun `train.py` to redownload.

### Reproducible training notes (what we changed in code)

- **Augmentation:** training pipeline uses `RandomCrop(32, padding=4)`, `RandomHorizontalFlip`, and light `ColorJitter` to improve generalization on CIFAR-10.
- **Normalization:** standard CIFAR mean/std are applied before training.
- **Initialization & normalization:** models use Kaiming (He) initialization and the ZFNet implementation adds `BatchNorm2d` after convolutions for more stable / faster convergence.
- **Optimizer & schedule:** the default is SGD with momentum (0.9) and a StepLR schedule; these tend to outperform Adam for standard CIFAR training when run for many epochs.
- **Logging:** training logs (batch/epoch loss and accuracy) are written to `checkpoints/runs` for inspection with TensorBoard.

### Commands

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python train.py --model lenet --epochs 10
.venv/bin/python train.py --model zfnet --epochs 10
```

### Results table (fill in after your runs)

| Model | Epochs | Best test accuracy | Train time (device) | Notes |
| --- | ---: | ---: | --- | --- |
| LeNet-5 | 10 | 99.09% | 198.1s (CPU) | MNIST, resized to 32×32 |
| ZFNet (CIFAR-adapted) |  |  |  |  |

**Completed LeNet run (CPU, 10 epochs, SGD defaults):** LeNet-5 reached **99.09%** best test accuracy on MNIST in **198.1s**.

---

## VII. Conclusion

The path from **LeNet-5** to **ZFNet** shows two coupled trends: **scale** (data, resolution, depth, hardware) and **scientific method** (using **Deconvnet-style probes** and controlled experiments to justify **specific layer changes**). ZFNet’s first-layer change (**11×11 stride 4 → 7×7 stride 2** on ImageNet-scale inputs) is the clearest “single slide” example of **visualization-driven architecture editing**. Our code reproduces the **historical stacking ideas** on CIFAR-10 while documenting every deliberate simplification above.

---

## References

1. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-Based Learning Applied to Document Recognition,” *Proc. IEEE*, 1998.  
2. A. Krizhevsky, I. Sutskever, and G. Hinton, “ImageNet Classification with Deep Convolutional Neural Networks,” NeurIPS, 2012.  
3. M. D. Zeiler and R. Fergus, “Visualizing and Understanding Convolutional Networks,” ECCV, 2014 (arXiv:1311.2901, 2013).  

---

## Slide / presentation checklist (optional)

- One figure: **AlexNet layer‑1** vs **ZFNet layer‑1** filter geometry (11×11/s4 vs 7×7/s2).  
- One panel: **Deconvnet** pipeline (**unpool → ReLU → filter**).  
- One table: **LeNet vs AlexNet vs ZFNet** (input size, classes, parameters order-of-magnitude, GPU need).  
- One honesty slide: **this repo** uses **CIFAR-10** and a **compact head**; full ImageNet ZFNet matches the paper’s FC and input resolution.
