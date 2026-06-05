import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


def build_model(num_classes: int = 10, freeze_backbone: bool = True) -> nn.Module:
    model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # replace the classifier head: (dropout + Linear(1280, 1000)) -> (dropout + Linear(1280, num_classes))
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
