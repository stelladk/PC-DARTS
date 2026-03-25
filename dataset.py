"""
dataset.py  -  flexible dataset loading for PC-DARTS  [UPDATED]
================================================================
Drop this file next to the PC-DARTS source files and import it from
train_search_custom.py / train_custom.py.

Supports
--------
* CIFAR-10 / CIFAR-100          (torchvision built-in, auto-download)
* ImageNet                       (standard torchvision ImageFolder layout)
* Any ImageFolder layout         (root/train/<class>/img.jpg)
* STL-10, SVHN, Flowers-102, Food-101
* Flat folder (no subdirs)       via FlatImageDataset
* NpyWebDataset                  numpy arrays downloaded from a URL (see below)
* Custom datasets                via the @register_dataset decorator

Quick API
---------
    from dataset import get_dataset
    train_data, n_classes, in_channels = get_dataset(args)

Using NpyWebDataset
-------------------
    # args.dataset = 'npyweb'
    # args.npyweb_url  = 'https://example.com/mydata.zip'
    # args.npyweb_name = 'mydata'          (optional, used as cache key)
    # args.npyweb_root = 'data/webdatasets/npy'  (optional)
    # args.npyweb_data_key  = '_x'         (optional)
    # args.npyweb_label_key = '_y'         (optional)

To compute per-channel stats for an unknown dataset:
    from dataset import compute_dataset_stats
    mean, std = compute_dataset_stats(train_data)
"""

import hashlib
import os
import zipfile
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image
from tools.augmentations import default_augmentations, get_transforms
from tools.augmentations import npy_datasets as _NPY_DATASETS_AUG
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

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

    def __call__(self, img):  # img: CxHxW tensor
        h, w = img.size(1), img.size(2)
        y = torch.randint(h, (1,)).item()
        x = torch.randint(w, (1,)).item()
        y1, y2 = max(0, y - self.length // 2), min(h, y + self.length // 2)
        x1, x2 = max(0, x - self.length // 2), min(w, x + self.length // 2)
        mask = torch.ones(h, w, dtype=img.dtype)
        mask[y1:y2, x1:x2] = 0.0
        return img * mask.unsqueeze(0)


# ──────────────────────────────────────────────────────── transforms ── #


def cifar_transforms(cutout=False, cutout_length=16, train=True):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    if train:
        t = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
        if cutout:
            t.append(Cutout(cutout_length))
    else:
        t = [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    return transforms.Compose(t)


def imagenet_transforms(image_size=224, train=True):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.4, 0.4, 0.4),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def generic_transforms(
    image_size=32,
    mean=None,
    std=None,
    train=True,
    cutout=False,
    cutout_length=16,
    grayscale=False,
):
    """Sensible defaults for arbitrary image classification datasets."""
    if mean is None:
        mean = [0.485, 0.456, 0.406]
    if std is None:
        std = [0.229, 0.224, 0.225]

    pre = [transforms.Grayscale(1)] if grayscale else []

    if train:
        if image_size <= 64:
            spatial = [
                transforms.RandomCrop(image_size, padding=image_size // 8),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            spatial = [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
            ]
        t = pre + spatial + [transforms.ToTensor(), transforms.Normalize(mean, std)]
        if cutout:
            t.append(Cutout(cutout_length))
    else:
        if image_size <= 64:
            spatial = [transforms.Resize(image_size)]
        else:
            spatial = [
                transforms.Resize(int(image_size * 256 / 224)),
                transforms.CenterCrop(image_size),
            ]
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

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.paths = [
            os.path.join(root, f)
            for f in sorted(os.listdir(root))
            if os.path.splitext(f)[1].lower() in self.EXTENSIONS
        ]
        if not self.paths:
            raise RuntimeError(f"No images found in {root}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, 0


# ──────────────────────────────────────── numpy-array transforms ────── #


class NumpyToTensor:
    """
    Convert a numpy array to a float32 torch.Tensor, handling all common
    array layouts that come out of NpyWebDataset:

      HxW        = 1xHxW   (greyscale, no channel dim)
      HxWxC      = CxHxW   (PIL-style, channels last)
      CxHxW      = CxHxW   (already channels first - pass through)
      N (1-D)    = (N,)     (feature vector / time series - no reshape)

    Values are cast to float32.  If the array is uint8 (0-255) it is
    scaled to [0, 1] automatically.
    """

    def __call__(self, x: np.ndarray) -> torch.Tensor:
        x = np.asarray(x, dtype=np.float32)
        if x.dtype == np.uint8 or x.max() > 1.0 + 1e-3:
            # only scale if we're clearly in [0,255] range
            x = x / 255.0
        if x.ndim == 2:  # HxW  = 1xHxW
            x = x[np.newaxis, :, :]
        elif x.ndim == 3 and x.shape[-1] in (1, 3, 4):  # HxWxC = CxHxW
            x = np.transpose(x, (2, 0, 1))
        # else: already CxHxW or 1-D feature vector - leave as-is
        return torch.from_numpy(x.copy())


class NumpyNormalize:
    """
    Normalise a CHxW float tensor channel-wise.
    Mirrors torchvision.transforms.Normalize but accepts arbitrary C.
    """

    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim < 3:
            return t  # feature vectors - skip normalisation
        return (t - self.mean.to(t.device)) / self.std.to(t.device)


class NumpyRandomHFlip:
    """Horizontal flip on a CxHxW tensor (50 % probability)."""

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim < 3:
            return t
        if torch.rand(1).item() < 0.5:
            return t.flip(-1)
        return t


class NumpyRandomCrop:
    """
    Random crop with reflection padding on a CxHxW tensor.
    Equivalent to transforms.RandomCrop(size, padding=padding).
    """

    def __init__(self, size, padding):
        self.size = size
        self.padding = padding

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim < 3:
            return t
        p = self.padding
        # pad: (left, right, top, bottom)
        padded = torch.nn.functional.pad(
            t.unsqueeze(0), (p, p, p, p), mode="reflect"
        ).squeeze(0)
        h, w = padded.shape[-2], padded.shape[-1]
        th, tw = self.size, self.size
        i = torch.randint(0, h - th + 1, (1,)).item()
        j = torch.randint(0, w - tw + 1, (1,)).item()
        return padded[..., i : i + th, j : j + tw]


def npy_transforms(
    image_size: int, mean, std, train: bool, cutout: bool = False, cutout_length: int = 16
) -> Callable:
    """
    Build a transform pipeline that works on raw numpy arrays
    (as returned by NpyWebDataset.__getitem__).

    The pipeline is:
        NumpyToTensor = (augmentations if train) = NumpyNormalize = (Cutout)
    """
    to_tensor = NumpyToTensor()
    normalize = NumpyNormalize(mean, std)

    if train:
        aug = [
            NumpyRandomCrop(image_size, padding=max(1, image_size // 8)),
            NumpyRandomHFlip(),
        ]
        steps = [to_tensor] + aug + [normalize]
        if cutout:
            steps.append(Cutout(cutout_length))
    else:
        steps = [to_tensor, normalize]

    def pipeline(x):
        for fn in steps:
            x = fn(x)
        return x

    return pipeline


# ───────────────────────────────────────────── NpyWebDataset ─────────── #


class NpyWebDataset(Dataset):
    """
    Downloads a ZIP of .npy files from a URL, extracts them, and serves
    (array, label) pairs.

    Expected ZIP layout
    -------------------
    Any nesting is fine.  Files are matched by name patterns:
      data   files: contain ``data_key``  (default ``_x``) AND a split prefix
      label  files: contain ``label_key`` (default ``_y``) AND a split prefix

    Split prefixes
    --------------
    train=True  =  looks for files whose name contains 'train' or 'valid'
    train=False =  looks for files whose name contains 'test'

    Example file names that work out of the box
    --------------------------------------------
      train_x.npy, train_y.npy
      train_data_x.npy, train_data_y.npy
      valid_x.npy, valid_y.npy
      test_x.npy,  test_y.npy

    Parameters
    ----------
    url        : str   - direct download URL of the ZIP file
    train      : bool  - True = train+valid split, False = test split
    root       : str   - local cache directory
    name       : str   - human-readable cache key (SHA-256 of URL if empty)
    download   : bool  - if True, loads everything into RAM; if False, uses
                         memory-mapped lazy loading
    transform  : callable | None - applied to each sample array
    data_key   : str   - substring that identifies data files (default '_x')
    label_key  : str   - substring that identifies label files (default '_y')
    """

    def __init__(
        self,
        url: str,
        train: bool = True,
        root: str = "data/webdatasets/npy",
        name: str = "",
        download: bool = True,
        transform: Optional[Callable] = None,
        data_key: str = "_x",
        label_key: str = "_y",
    ):
        self.url = url
        self.name = name
        self.train = train
        self.root = Path(os.path.expanduser(root))
        self.download = download
        self.transform = transform
        self.data_key = data_key
        self.label_key = label_key

        self.local_zip_path = self._download_and_extract()
        self.data_files, self.label_files = self._find_data_and_labels()
        self.data, self.labels = self._load_data() if download else (None, None)

    @property
    def targets(self) -> Optional[np.ndarray]:
        return self.labels

    # ------------------------------------------------------------------ #
    # internal helpers                                                      #
    # ------------------------------------------------------------------ #

    def _download_and_extract(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        if self.name == "":
            hash_name = hashlib.sha256(self.url.encode()).hexdigest()
        else:
            hash_name = self.name
        zip_path = self.root / f"{hash_name}.zip"
        extract_dir = self.root / f"{hash_name}_extracted"

        if not zip_path.exists():
            r = requests.get(self.url)
            r.raise_for_status()
            with open(zip_path, "wb") as f:
                f.write(r.content)

        if not extract_dir.exists():
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

        return extract_dir

    def _find_data_and_labels(self) -> Tuple[List[Path], List[Path]]:
        files = list(self.local_zip_path.rglob("*.npy"))
        prefix = ["train", "valid"] if self.train else ["test"]

        data_files = sorted(
            [
                f
                for f in files
                if (self.data_key in f.name) and any(ix in f.name for ix in prefix)
            ]
        )
        label_files = sorted(
            [
                f
                for f in files
                if (self.label_key in f.name) and any(ix in f.name for ix in prefix)
            ]
        )

        assert len(data_files) == len(label_files), (
            f"Mismatch: {len(data_files)} data files vs "
            f"{len(label_files)} label files"
        )
        assert len(data_files) > 0, (
            f"No .npy files found for prefixes {prefix} "
            f"with data_key='{self.data_key}' in {self.local_zip_path}"
        )
        return data_files, label_files

    def _load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = np.concatenate([np.load(f) for f in self.data_files], axis=0)
        labels = np.concatenate([np.load(f) for f in self.label_files], axis=0)
        return data, labels

    def _resolve_index(self, index: int) -> Tuple[int, int]:
        """Map a global index to (file_idx, local_idx)."""
        cumulative = 0
        for i, f in enumerate(self.data_files):
            n = np.load(f, mmap_mode="r").shape[0]
            if index < cumulative + n:
                return i, index - cumulative
            cumulative += n
        raise IndexError(f"Index {index} out of bounds (dataset size {cumulative})")

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                      #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        if self.download:
            assert self.data is not None
            return len(self.data)
        total = sum(np.load(f, mmap_mode="r").shape[0] for f in self.data_files)
        return total

    def __getitem__(self, index: int):
        if self.download:
            assert self.data is not None and self.labels is not None
            x, y = self.data[index], self.labels[index]
        else:
            file_idx, local_idx = self._resolve_index(index)
            x = np.load(self.data_files[file_idx])[local_idx]
            y = np.load(self.label_files[file_idx])[local_idx]

        if self.transform is not None:
            x = self.transform(x)
        return x, int(y)

    # ------------------------------------------------------------------ #
    # Convenience                                                           #
    # ------------------------------------------------------------------ #

    @property
    def shape(self) -> Optional[Tuple]:
        """Shape of a single sample array (without batch dim)."""
        if self.data is not None:
            return self.data.shape[1:]
        # peek at first file
        return np.load(self.data_files[0], mmap_mode="r").shape[1:]

    def infer_image_meta(self) -> Tuple[int, int]:
        """
        Return (in_channels, image_size) by inspecting the sample shape.

        Handles:
          (C, H, W)  =  (C, H)
          (H, W, C)  =  (C, H)
          (H, W)     =  (1, H)
          (N,)       =  (1, N)   feature vector - image_size = N
        """
        s = self.shape
        if s is None:
            return 3, 32
        if len(s) == 3:
            # pick the layout by checking which axis is smallest
            if s[0] <= 4:  # (C, H, W)
                return s[0], s[1]
            elif s[2] <= 4:  # (H, W, C)
                return s[2], s[0]
            else:  # ambiguous - assume (C,H,W)
                return s[0], s[1]
        elif len(s) == 2:  # (H, W) greyscale
            return 1, s[0]
        else:  # 1-D feature vector
            return 1, s[0]


# ─────────────────────────────────────────── built-in factories ──────── #


@register_dataset("cifar10")
def _(root, train, transform):
    return datasets.CIFAR10(root=root, train=train, download=True, transform=transform)


@register_dataset("cifar100")
def _(root, train, transform):
    return datasets.CIFAR100(root=root, train=train, download=True, transform=transform)


@register_dataset("imagenet")
def _(root, train, transform):
    split = "train" if train else "val"
    return datasets.ImageFolder(os.path.join(root, split), transform=transform)


@register_dataset("imagefolder")
def _(root, train, transform):
    """
    Tries root/train or root/val first; falls back to root itself.
    The fallback is convenient when you pass --train_portion to split on-the-fly.
    """
    split = "train" if train else "val"
    split_path = os.path.join(root, split)
    if os.path.isdir(split_path):
        return datasets.ImageFolder(split_path, transform=transform)
    return datasets.ImageFolder(root, transform=transform)


@register_dataset("stl10")
def _(root, train, transform):
    return datasets.STL10(
        root=root, split="train" if train else "test", download=True, transform=transform
    )


@register_dataset("svhn")
def _(root, train, transform):
    return datasets.SVHN(
        root=root, split="train" if train else "test", download=True, transform=transform
    )


@register_dataset("flowers102")
def _(root, train, transform):
    return datasets.Flowers102(
        root=root, split="train" if train else "val", download=True, transform=transform
    )


@register_dataset("food101")
def _(root, train, transform):
    return datasets.Food101(
        root=root, split="train" if train else "test", download=True, transform=transform
    )


@register_dataset("flat")
def _(root, train, transform):
    """All images in a single flat folder with no class subdirs."""
    return FlatImageDataset(root=root, transform=transform)


@register_dataset("addnist")
def _(root, train, transform):
    """
    NpyWebDataset - numpy arrays downloaded from a URL.

    The factory ignores `root` (the cache directory comes from
    args.npyweb_root) and `transform` (built separately in get_dataset
    because we need the sample shape first).  Both are accepted so the
    signature stays compatible with the registry protocol; the real
    construction happens inside get_dataset() / get_val_dataset().

    You should not call this factory directly - use get_dataset(args).
    """
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/24574354/versions/2",
        name="AddNIST",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


@register_dataset("multnist")
def _(root, train, transform):
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/24574678/versions/2",
        name="MultNIST",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


@register_dataset("cifartile")
def _(root, train, transform):
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/24551539/versions/1",
        name="CIFARTile",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    if dataset.data.ndim == 4:
        dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


@register_dataset("language")
def _(root, train, transform):
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/24574729/versions/1",
        name="LanguageASPELL",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    if dataset.data.ndim == 4:
        dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


@register_dataset("gutenberg")
def _(root, train, transform):
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/24574753/versions/1",
        name="Gutenberg",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    if dataset.data.ndim == 4:
        dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


@register_dataset("geoclassing")
def _(root, train, transform):
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/24050256/versions/3",
        name="GeoClassing",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    if dataset.data.ndim == 4:
        dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


@register_dataset("chesseract")
def _(root, train, transform):
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/24118743/versions/2",
        name="Chesseract",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    if dataset.data.ndim == 4:
        dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


@register_dataset("gameoflife")
def _(root, train, transform):
    dataset = NpyWebDataset(
        url="https://data.ncl.ac.uk/ndownloader/articles/30000835/versions/1",
        name="GameOfLife",
        train=train,
        root=root,
        download=True,
        transform=transform,
    )
    if dataset.data.ndim == 4:
        dataset.data = dataset.data.transpose(0, 2, 3, 1)
    return dataset


# ───────────────────────────────────────────────── metadata table ──────── #

# Datasets backed by NpyWebDataset - sourced from tools.augmentations.
NPY_DATASETS = _NPY_DATASETS_AUG

# (n_classes, default_image_size, in_channels)
# n_classes=None means infer at runtime from dataset.classes
DATASET_META = {
    "cifar10": (10, 32, 3),
    "cifar100": (100, 32, 3),
    "imagenet": (1000, 224, 3),
    "imagefolder": (None, 224, 3),
    "stl10": (10, 96, 3),
    "svhn": (10, 32, 3),
    "flowers102": (102, 224, 3),
    "food101": (101, 224, 3),
    "flat": (1, 32, 3),
    "addnist": (20, 28, 3),
    "multnist": (10, 28, 3),
    "cifartile": (4, 32, 3),
    "language": (10, 128, 1),
    "gutenberg": (6, 128, 1),
    "geoclassing": (10, 64, 3),
    "chesseract": (3, 32, 12),
    "gameoflife": (25, 32, 1),
}


# ─────────────────────────────────────────────────────── public API ──── #


def get_dataset(args):
    """
    Build the training dataset (and infer metadata) from args.

    Required args attributes
    ------------------------
    args.dataset  : str   - key in DATASET_REGISTRY
    args.data     : str   - path to dataset root

    Optional args attributes
    ------------------------
    args.image_size         : int
    args.num_classes        : int        override auto-detected value
    args.cutout             : bool
    args.cutout_length      : int
    args.no_augment         : bool       disable all training-time augmentation
    args.grayscale          : bool
    args.dataset_mean       : list[float]
    args.dataset_std        : list[float]
    args.data_augmentation  : list[str]  override default_augmentations for this dataset

    Returns
    -------
    train_data  : torch.utils.data.Dataset
    n_classes   : int
    in_channels : int
    """
    name = args.dataset.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{name}'. " f"Registered: {sorted(DATASET_REGISTRY)}"
        )

    meta = DATASET_META.get(name, (None, 224, 3))
    n_classes_meta = meta[0]
    default_size = meta[1]
    in_channels = meta[2]

    image_size = getattr(args, "image_size", None) or default_size
    no_augment = getattr(args, "no_augment", False)
    cutout = getattr(args, "cutout", False) and not no_augment
    cutout_len = getattr(args, "cutout_length", 16)
    grayscale = getattr(args, "grayscale", False)
    mean = getattr(args, "dataset_mean", None)
    std = getattr(args, "dataset_std", None)

    if grayscale:
        in_channels = 1

    # Determine augmentation list: args override > default_augmentations > None (legacy)
    if no_augment:
        aug_list: Optional[List[str]] = []
    else:
        aug_list = getattr(args, "data_augmentation", None)
        if aug_list is None:
            aug_list = default_augmentations.get(name, None)

    # build transform
    if aug_list is not None:
        # Use shared get_transforms from tools.augmentations
        base_t, aug_t = get_transforms(name, aug_list)
        if name in NPY_DATASETS:
            steps = base_t + aug_t
        else:
            steps = aug_t + base_t
        if cutout:
            steps.append(Cutout(cutout_len))
        transform = transforms.Compose(steps)
    elif name == "imagenet":
        transform = imagenet_transforms(image_size=image_size, train=True)
    elif name in NPY_DATASETS:
        transform = npy_transforms(
            image_size=image_size,
            mean=mean or [0.5] * in_channels,
            std=std or [0.5] * in_channels,
            train=True,
            cutout=cutout,
            cutout_length=cutout_len,
        )
    else:
        transform = generic_transforms(
            image_size=image_size,
            mean=mean,
            std=std,
            train=True,
            cutout=cutout,
            cutout_length=cutout_len,
            grayscale=grayscale,
        )

    # Build base (no-augmentation) transforms for the test split
    if aug_list is not None:
        base_t, _ = get_transforms(name, [])
        base_transform = transforms.Compose(base_t) if base_t else None
    elif name == "imagenet":
        base_transform = imagenet_transforms(image_size=image_size, train=False)
    elif name in NPY_DATASETS:
        base_transform = npy_transforms(
            image_size=image_size,
            mean=mean or [0.5] * in_channels,
            std=std or [0.5] * in_channels,
            train=False,
        )
    else:
        base_transform = generic_transforms(
            image_size=image_size, mean=mean, std=std, train=False, grayscale=grayscale
        )

    train_data = DATASET_REGISTRY[name](root=args.data, train=True, transform=transform)
    test_data = DATASET_REGISTRY[name](root=args.data, train=False, transform=base_transform)

    # infer n_classes
    if n_classes_meta is not None:
        n_classes = n_classes_meta
    elif hasattr(train_data, "classes"):
        n_classes = len(train_data.classes)
    elif hasattr(train_data, "targets"):
        n_classes = len(set(train_data.targets))
    else:
        raise RuntimeError(
            "Cannot infer n_classes automatically. " "Pass --num_classes explicitly."
        )

    # allow CLI override
    if getattr(args, "num_classes", None):
        n_classes = args.num_classes

    return train_data, test_data, n_classes, in_channels


def get_val_dataset(args):
    """Same as get_dataset but returns the validation/test split."""
    name = args.dataset.lower()
    if name not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'.")

    meta = DATASET_META.get(name, (None, 224, 3))
    default_size = meta[1]
    image_size = getattr(args, "image_size", None) or default_size
    grayscale = getattr(args, "grayscale", False)
    mean = getattr(args, "dataset_mean", None)
    std = getattr(args, "dataset_std", None)

    in_channels = DATASET_META.get(name, (None, 224, 3))[2]
    # Validation always uses base transforms only (no augmentation)
    if name in default_augmentations:
        base_t, _ = get_transforms(name, [])
        transform = transforms.Compose(base_t)
    elif name == "imagenet":
        transform = imagenet_transforms(image_size=image_size, train=False)
    elif name in NPY_DATASETS:
        transform = npy_transforms(
            image_size=image_size,
            mean=mean or [0.5] * in_channels,
            std=std or [0.5] * in_channels,
            train=False,
        )
    else:
        transform = generic_transforms(
            image_size=image_size, mean=mean, std=std, train=False, grayscale=grayscale
        )

    return DATASET_REGISTRY[name](root=args.data, train=False, transform=transform)


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
    loader = DataLoader(
        torch.utils.data.Subset(dataset, indices.tolist()),
        batch_size=256,
        shuffle=False,
        num_workers=4,
    )

    # Use a plain tensor transform dataset for computation
    mean = torch.zeros(3)
    std = torch.zeros(3)
    n = 0
    for imgs, _ in loader:
        b = imgs.size(0)
        mean += imgs.mean(dim=(0, 2, 3)) * b
        std += imgs.std(dim=(0, 2, 3)) * b
        n += b
    return (mean / n).tolist(), (std / n).tolist()
