# PC-DARTS Custom Dataset Guide

This guide explains how to run [PC-DARTS](https://github.com/yuhuixu1993/PC-DARTS)
on **any image classification dataset**, not just CIFAR-10 and ImageNet.

Two files are provided:

| File | Purpose |
|------|---------|
| `dataset.py` | Dataset registry + transforms + `get_dataset()` API |
| `train_search_custom.py` | Drop-in for `train_search.py` with `--dataset` flag |

---

## 1. Installation

```bash
git clone https://github.com/yuhuixu1993/PC-DARTS
cd PC-DARTS
# copy both files into the repo root
cp dataset.py train_search_custom.py .
pip install torch torchvision
```

---

## 2. Built-in datasets

| `--dataset` | Classes | Default size | Notes |
|------------|---------|-------------|-------|
| `cifar10` | 10 | 32 | Auto-download |
| `cifar100` | 100 | 32 | Auto-download |
| `imagenet` | 1000 | 224 | Needs manual download |
| `imagefolder` | auto | 224 | See §3 |
| `stl10` | 10 | 96 | Auto-download |
| `svhn` | 10 | 32 | Auto-download |
| `flowers102` | 102 | 224 | Auto-download |
| `food101` | 101 | 224 | Auto-download |

### CIFAR-10 (identical to original)
```bash
python train_search_custom.py --data /data/cifar
```

### CIFAR-100
```bash
python train_search_custom.py \
    --dataset cifar100 \
    --data /data/cifar \
    --epochs 50
```

### STL-10
```bash
python train_search_custom.py \
    --dataset stl10 \
    --data /data/stl10 \
    --image_size 96 \
    --batch_size 64
```

### Flowers-102
```bash
python train_search_custom.py \
    --dataset flowers102 \
    --data /data/flowers \
    --image_size 224 \
    --batch_size 32 \
    --train_portion 0.8
```

---

## 3. Your own dataset (ImageFolder layout)

Organize your data as:

```
/data/mydata/
    train/
        cat/  img001.jpg  img002.jpg ...
        dog/  img003.jpg  ...
    val/
        cat/  ...
        dog/  ...
```

Then run:

```bash
python train_search_custom.py \
    --dataset imagefolder \
    --data /data/mydata \
    --image_size 64 \
    --batch_size 128
```

The number of classes is **auto-detected** from the folder structure.

If you only have one split (no `train/` sub-directory):

```
/data/mydata/
    cat/  img001.jpg ...
    dog/  img002.jpg ...
```

Use `--train_portion 0.5` to split on-the-fly (the default).

---

## 4. Custom per-channel normalization stats

If your dataset has very different statistics from ImageNet, compute them first:

```python
# compute_stats.py
import argparse
from torchvision import transforms
from torchvision.datasets import ImageFolder
from dataset import compute_dataset_stats

ds = ImageFolder("/data/mydata/train",
                 transform=transforms.ToTensor())
mean, std = compute_dataset_stats(ds, n_samples=10000)
print(f"--dataset_mean {mean[0]:.4f} {mean[1]:.4f} {mean[2]:.4f}")
print(f"--dataset_std  {std[0]:.4f} {std[1]:.4f} {std[2]:.4f}")
```

Then pass those values:

```bash
python train_search_custom.py \
    --dataset imagefolder \
    --data /data/mydata \
    --dataset_mean 0.45 0.42 0.38 \
    --dataset_std  0.22 0.21 0.23
```

---

## 5. Greyscale datasets

```bash
python train_search_custom.py \
    --dataset imagefolder \
    --data /data/xray \
    --grayscale \
    --image_size 64
```

**Important:** You also need to patch `model_search.py` to accept `in_channels`
(see §7).

---

## 6. Registering a fully custom dataset

Add this to `dataset.py` (or a separate file you import):

```python
from dataset import register_dataset
from torch.utils.data import Dataset

class MySpecialDataset(Dataset):
    def __init__(self, root, train, transform):
        ...  # your loading logic

@register_dataset("mydata")
def _(root, train, transform):
    return MySpecialDataset(root, train=train, transform=transform)
```

Then use `--dataset mydata`.

---

## 7. Patching `model_search.py` for dynamic class count

The original `model_search.py` has `CIFAR_CLASSES = 10` hard-coded.
`train_search_custom.py` already passes `n_classes` as the second argument to
`Network(...)`, which matches the constructor signature — **no patch needed for
class count**.

However, if you use `--grayscale` (1 input channel instead of 3), you need a
one-line patch in `model_search.py`:

```python
# Original line (~line 90):
self.stem = nn.Sequential(
    nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
    ...
)

# Patched (add in_channels=3 argument):
self.stem = nn.Sequential(
    nn.Conv2d(in_channels, C_curr, 3, padding=1, bias=False),
    ...
)
```

And update the `Network.__init__` signature:

```python
def __init__(self, C, num_classes, layers, criterion, k=4, in_channels=3):
    ...
    self.in_channels = in_channels
```

---

## 8. Key arguments reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `cifar10` | Dataset name |
| `--data` | `../data` | Dataset root path |
| `--image_size` | auto | Override spatial resolution |
| `--num_classes` | auto | Override class count |
| `--dataset_mean R G B` | ImageNet | Per-channel mean |
| `--dataset_std R G B` | ImageNet | Per-channel std |
| `--grayscale` | off | Convert to single-channel |
| `--train_portion` | 0.5 | Fraction used for weights (rest for arch) |
| `--batch_size` | 256 | Batch size |
| `--image_size` | dataset default | Spatial resolution |
| `--k` | 4 | PC-DARTS partial channel ratio |
| `--epochs` | 50 | Search epochs |

---

## 9. Tips for new datasets

- **Small datasets (< 5 k images):** increase `--train_portion` to 0.8 so the
  weight network sees more data.
- **High-resolution inputs:** start with `--image_size 64` to keep GPU memory
  under control during search, then retrain the found genotype at full
  resolution.
- **Many classes (> 100):** increase `--init_channels` to 32 and `--layers` to
  14 for better capacity.
- **Very few classes (binary):** the default settings work fine; consider
  reducing `--epochs` to 20-30.