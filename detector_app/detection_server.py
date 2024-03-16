import requests
from PIL import Image
from io import BytesIO

import torch
from torchvision import transforms
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn_v2,
    MaskRCNN_ResNet50_FPN_V2_Weights
)
from prometheus_client import Counter
from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics


app = Flask(__name__, static_url_path="")
metrics = PrometheusMetrics(app)


def _setup_model():
    model = maskrcnn_resnet50_fpn_v2(
        weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    )
    model.eval()
    return model

to_tensor = transforms.Compose([transforms.PILToTensor()])
labels = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
model = _setup_model()


def _infer(image):
    with torch.no_grad():
        tensor = to_tensor(image)
        tensor = tensor / 255 * 2 - 1
        result = model([tensor])
    return result[0]["labels"]

@app.route("/predict", methods=['POST'])
@metrics.counter("app_http_inference_count", "Number of inferencs")
@metrics.counter("api_invocations_total", "Number of endpoints")
def predict():
    data = request.get_json(force=True)

    response = requests.get(data['url'])
    image = Image.open(BytesIO(response.content))

    result = _infer(image)

    return jsonify({
        "objects": [labels[idx] for idx in result.tolist()]
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
