"""
dataset.py  —  flexible dataset loading for PC-DARTS
=====================================================
Drop this file next to the PC-DARTS source files and import it from
train_search_custom.py / train_custom.py.

Supports
--------
* CIFAR-10 / CIFAR-100          (torchvision built-in, auto-download)
* ImageNet                       (standard torchvision ImageFolder layout)
* Any ImageFolder layout         (root/train/<class>/img.jpg)
* STL-10, SVHN, Flowers-102, Food-101
* Flat folder (no subdirs)       via FlatImageDataset
* Custom datasets                via the @register_dataset decorator

Quick API
---------
    from dataset import get_dataset
    train_data, n_classes, in_channels = get_dataset(args)

To compute per-channel stats for an unknown dataset:
    from dataset import compute_dataset_stats
    mean, std = compute_dataset_stats(train_data)
"""

import os
import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

# ──────────────────────────────────────────────────────── registry ──── #

DATASET_REGISTRY = {}


def register_dataset(name):
    """Decorator to add a custom dataset factory to the registry.

    The factory signature must be:
        fn(root: str, train: bool, transform) -> Dataset

    Example
    -------
    @register_dataset("my_dataset")
    def _(root, train, transform):
        return MyDataset(root, split="train" if train else "val",
                         transform=transform)
    """
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


# ─────────────────────────────────────────────────── augmentations ──── #

class Cutout:
    """Randomly masks one square patch (Devries & Taylor, 2017)."""
    def __init__(self, length):
        self.length = length

    def __call__(self, img):           # img: C×H×W tensor
        h, w = img.size(1), img.size(2)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        mask = torch.ones(h, w, dtype=img.dtype)
        mask[y1:y2, x1:x2] = 0.
        return img * mask.unsqueeze(0)


# ──────────────────────────────────────────────────────── transforms ── #

def cifar_transforms(cutout=False, cutout_length=16, train=True):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD  = [0.24703233, 0.24348505, 0.26158768]
    if train:
        t = [transforms.RandomCrop(32, padding=4),
             transforms.RandomHorizontalFlip(),
             transforms.ToTensor(),
             transforms.Normalize(MEAN, STD)]
        if cutout:
            t.append(Cutout(cutout_length))
    else:
        t = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


def imagenet_transforms(image_size=224, train=True):
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4, 0.4, 0.4),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def generic_transforms(image_size=32, mean=None, std=None, train=True,
                       cutout=False, cutout_length=16, grayscale=False):
    """Sensible defaults for arbitrary image classification datasets."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std  = [0.229, 0.224, 0.225]

    pre = [transforms.Grayscale(1)] if grayscale else []

    if train:
        if image_size <= 64:
            spatial = [transforms.RandomCrop(image_size, padding=image_size // 8),
                       transforms.RandomHorizontalFlip()]
        else:
            spatial = [transforms.RandomResizedCrop(image_size),
                       transforms.RandomHorizontalFlip()]
        t = pre + spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout:
            t.append(Cutout(cutout_length))
    else:
        if image_size <= 64:
            spatial = [transforms.Resize(image_size)]
        else:
            spatial = [transforms.Resize(int(image_size * 256 / 224)),
                       transforms.CenterCrop(image_size)]
        t = pre + spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)]

    return transforms.Compose(t)


# ──────────────────────────────────────────── flat-folder dataset ────── #

class FlatImageDataset(Dataset):
    """
    All images live directly in root/ (no class subdirectories).
    All samples are assigned label 0.  Useful for single-class or unlabelled
    datasets during NAS search.

    root/
      img001.jpg
      img002.png
      ...
    """
    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    def __init__(self, root, transform=None):
        self.root      = root
        self.transform = transform
        self.paths     = [
            os.path.join(root, f) for f in sorted(os.listdir(root))
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, 0


# ─────────────────────────────────────────── built-in factories ──────── #

@register_dataset('cifar10')
def _(root, train, transform):
    return datasets.CIFAR10(root=root, train=train, download=True,
                            transform=transform)


@register_dataset('cifar100')
def _(root, train, transform):
    return datasets.CIFAR100(root=root, train=train, download=True,
                             transform=transform)


@register_dataset('imagenet')
def _(root, train, transform):
    split = 'train' if train else 'val'
    return datasets.ImageFolder(os.path.join(root, split), transform=transform)


@register_dataset('imagefolder')
def _(root, train, transform):
    """
    Tries root/train or root/val first; falls back to root itself.
    The fallback is convenient when you pass --train_portion to split on-the-fly.
    """
    split      = 'train' if train else 'val'
    split_path = os.path.join(root, split)
    if os.path.isdir(split_path):
        return datasets.ImageFolder(split_path, transform=transform)
    return datasets.ImageFolder(root, transform=transform)


@register_dataset('stl10')
def _(root, train, transform):
    return datasets.STL10(root=root, split='train' if train else 'test',
                          download=True, transform=transform)


@register_dataset('svhn')
def _(root, train, transform):
    return datasets.SVHN(root=root, split='train' if train else 'test',
                         download=True, transform=transform)


@register_dataset('flowers102')
def _(root, train, transform):
    return datasets.Flowers102(root=root, split='train' if train else 'val',
                               download=True, transform=transform)


@register_dataset('food101')
def _(root, train, transform):
    return datasets.Food101(root=root, split='train' if train else 'test',
                            download=True, transform=transform)


@register_dataset('flat')
def _(root, train, transform):
    """All images in a single flat folder with no class subdirs."""
    return FlatImageDataset(root=root, transform=transform)


# ───────────────────────────────────────────────── metadata table ──────── #

# (n_classes, default_image_size, in_channels)
# n_classes=None means infer at runtime from dataset.classes
DATASET_META = {
    'cifar10'    : (10,   32,  3),
    'cifar100'   : (100,  32,  3),
    'imagenet'   : (1000, 224, 3),
    'imagefolder': (None, 224, 3),
    'stl10'      : (10,   96,  3),
    'svhn'       : (10,   32,  3),
    'flowers102' : (102,  224, 3),
    'food101'    : (101,  224, 3),
    'flat'       : (1,    32,  3),
}


# ─────────────────────────────────────────────────────── public API ──── #

def get_dataset(args):
    """
    Build the training dataset (and infer metadata) from args.

    Required args attributes
    ------------------------
    args.dataset  : str   — key in DATASET_REGISTRY
    args.data     : str   — path to dataset root

    Optional args attributes
    ------------------------
    args.image_size    : int
    args.num_classes   : int        override auto-detected value
    args.cutout        : bool
    args.cutout_length : int
    args.grayscale     : bool
    args.dataset_mean  : list[float]
    args.dataset_std   : list[float]

    Returns
    -------
    train_data  : torch.utils.data.Dataset
    n_classes   : int
    in_channels : int
    """
    name = args.dataset.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. "
            f"Registered: {sorted(DATASET_REGISTRY)}"
        )

    meta            = DATASET_META.get(name, (None, 224, 3))
    n_classes_meta  = meta[0]
    default_size    = meta[1]
    in_channels     = meta[2]

    image_size  = getattr(args, 'image_size',    None) or default_size
    cutout      = getattr(args, 'cutout',         False)
    cutout_len  = getattr(args, 'cutout_length',  16)
    grayscale   = getattr(args, 'grayscale',      False)
    mean        = getattr(args, 'dataset_mean',   None)
    std         = getattr(args, 'dataset_std',    None)

    if grayscale:
        in_channels = 1

    # build transform
    if name in ('cifar10', 'cifar100'):
        transform = cifar_transforms(cutout=cutout, cutout_length=cutout_len,
                                     train=True)
    elif name == 'imagenet':
        transform = imagenet_transforms(image_size=image_size, train=True)
    else:
        transform = generic_transforms(
            image_size=image_size, mean=mean, std=std,
            train=True, cutout=cutout, cutout_length=cutout_len,
            grayscale=grayscale)

    train_data = DATASET_REGISTRY[name](root=args.data, train=True,
                                        transform=transform)

    # infer n_classes
    if n_classes_meta is not None:
        n_classes = n_classes_meta
    elif hasattr(train_data, 'classes'):
        n_classes = len(train_data.classes)
    elif hasattr(train_data, 'targets'):
        n_classes = len(set(train_data.targets))
    else:
        raise RuntimeError(
            "Cannot infer n_classes automatically. "
            "Pass --num_classes explicitly."
        )

    # allow CLI override
    if getattr(args, 'num_classes', None):
        n_classes = args.num_classes

    return train_data, n_classes, in_channels


def get_val_dataset(args):
    """Same as get_dataset but returns the validation/test split."""
    name = args.dataset.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'.")

    meta         = DATASET_META.get(name, (None, 224, 3))
    default_size = meta[1]
    image_size   = getattr(args, 'image_size', None) or default_size
    grayscale    = getattr(args, 'grayscale',  False)
    mean         = getattr(args, 'dataset_mean', None)
    std          = getattr(args, 'dataset_std',  None)

    if name in ('cifar10', 'cifar100'):
        transform = cifar_transforms(train=False)
    elif name == 'imagenet':
        transform = imagenet_transforms(image_size=image_size, train=False)
    else:
        transform = generic_transforms(
            image_size=image_size, mean=mean, std=std,
            train=False, grayscale=grayscale)

    return DATASET_REGISTRY[name](root=args.data, train=False,
                                  transform=transform)


def compute_dataset_stats(dataset, n_samples=5000):
    """
    Estimate per-channel mean and std from a random subset of a dataset.
    Use this when you don't know the normalisation stats for a new dataset.

    Parameters
    ----------
    dataset   : torch.utils.data.Dataset  (images must already be tensors)
    n_samples : int

    Returns
    -------
    mean : list[float]   e.g. [0.45, 0.43, 0.39]
    std  : list[float]
    """
    indices = torch.randperm(len(dataset))[:n_samples]
    loader  = DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=256, shuffle=False, num_workers=4)

    # Use a plain tensor transform dataset for computation
    mean = torch.zeros(3)
    std  = torch.zeros(3)
    n    = 0
    for imgs, _ in loader:
        b     = imgs.size(0)
        mean += imgs.mean(dim=(0, 2, 3)) * b
        std  += imgs.std(dim=(0, 2, 3))  * b
        n    += b
    return (mean / n).tolist(), (std / n).tolist()