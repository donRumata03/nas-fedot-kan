"""
Copied from torchvision v0.12 (only v0.11 is available on the server).
Patched for in-gpu-memory caching
"""
import os
from pathlib import Path
from typing import Callable, Optional, Union

import torch
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive


class GpuCachedImageFolder(ImageFolder):
    def __init__(self, root, transform, target_transform: Optional[Callable] = None, eager: bool = False):
        super(GpuCachedImageFolder, self).__init__(root, transform=transform, target_transform=target_transform)
        self.cache = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if eager:
            print("Caching all images into GPU…")
            for i in range(len(self)):
                _img = self[i]
            print("DONE caching all images into GPU")

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]

        # Get the image and label using the parent class method
        image, label = super(GpuCachedImageFolder, self).__getitem__(index)
        image = image.to(self.device)

        # Cache the image and label
        self.cache[index] = (image, label)
        return image, label


class EuroSAT(GpuCachedImageFolder):
    """RGB version of the `EuroSAT <https://github.com/phelber/eurosat>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``root/eurosat`` exists.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If True, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again. Default is False.
    """

    def __init__(
            self,
            root: Union[str, Path],
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            eager: bool = False
    ) -> None:
        self.root = os.path.expanduser(root)
        self._base_folder = os.path.join(self.root, "eurosat")
        self._data_folder = os.path.join(self._base_folder, "2750")

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it")

        super().__init__(self._data_folder, transform=transform, target_transform=target_transform, eager=eager)
        self.root = os.path.expanduser(root)

    def __len__(self) -> int:
        return len(self.samples)

    def _check_exists(self) -> bool:
        return os.path.exists(self._data_folder)

    def download(self) -> None:

        if self._check_exists():
            return

        os.makedirs(self._base_folder, exist_ok=True)
        download_and_extract_archive(
            "https://madm.dfki.de/files/sentinel/EuroSAT.zip",
            download_root=self._base_folder,
            md5="c8fa014336c82ac7804f0398fcb19387",
        )
