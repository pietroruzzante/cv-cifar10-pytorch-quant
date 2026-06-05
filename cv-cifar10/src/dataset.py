import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize(224),
            T.RandomHorizontalFlip(),
            T.RandomCrop(224, padding=8),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ])
    return T.Compose([
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_dataloaders(data_dir: str, batch_size: int = 64, val_split: float = 0.1, num_workers: int = 2):
    train_full = CIFAR10(root=data_dir, train=True,  download=True, transform=get_transforms(train=True))
    test_set   = CIFAR10(root=data_dir, train=False, download=True, transform=get_transforms(train=False))

    val_size   = int(len(train_full) * val_split)
    train_size = len(train_full) - val_size
    train_set, val_set = random_split(train_full, [train_size, val_size])

    # val set uses inference transforms, not augmentation
    val_set.dataset.transform = get_transforms(train=False)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
