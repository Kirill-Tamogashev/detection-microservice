import requests
from PIL import Image
from io import BytesIO

import torch
from torchvision import transforms
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    # MaskRCNN_ResNet50_FPN_Weights,
    MaskRCNN_ResNet50_FPN_V2_Weights
)


def configure_inference_setup():
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, rpn_score_thresh=0.75)
    model.eval()

    to_tensor = transforms.Compose([
        transforms.PILToTensor(),
    ])
    labels = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
    return model, to_tensor, labels


def infer(url, model, transform, labels):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    with torch.no_grad():
        tensor = transform(image)
        tensor = tensor / 255
        result = model([tensor])

    return [labels[idx] for idx in result[0]["labels"].tolist()]