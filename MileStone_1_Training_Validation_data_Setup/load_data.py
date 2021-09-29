
from typing import Optional, Callable

from torchvision.transforms import AutoAugmentPolicy, AutoAugment, RandomApply, Compose, ToTensor, Normalize
from torchvision.datasets import ImageFolder


def load_data_from_ImageFolder(root: str, transform: Optional[Callable] = None):
    """Takes path of the root directory for the images
    creates a dataset using pytorch ImageFolder class
    using AutoAugment for CIFAR,IMAGENET, SVHN with probability 
    p=[0.2, 0.4, 0.4] if transform is not provided.

    Args:
        root (str): path to the diroctroy where the images are separated into their classes by their folder
        transform (Optional): Applies transform for the images to increase the variaty.
    Return:
        ImageFolder dataset.

    """
    if transform is None:
        polices = [AutoAugmentPolicy.CIFAR10,
                   AutoAugmentPolicy.IMAGENET, AutoAugmentPolicy.SVHN]
        augments = [AutoAugment(policy) for policy in polices]
        applier = RandomApply(transforms=augments, p=0.7)
        composed = Compose([applier, ToTensor(), Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    else:
        applier = transform
    dataset = ImageFolder(root=root, transform=composed)
    return dataset
