from flask import Flask, request, jsonify
from prometheus_flask_exporter import PrometheusMetrics

from inference_utils import (
    configure_inference_setup,
    infer,
)

app = Flask(__name__, static_url_path="")
metrics = PrometheusMetrics(app)
model, transform, labels = configure_inference_setup()


@app.route("/predict", methods=['POST'])
@metrics.counter("app_http_inference_count", "Number of inferencs")
@metrics.counter("api_invocations_total", "Number of endpoints")
def predict():
    data = request.get_json(force=True)
    result = infer(data['url'], model, transform, labels)

    return jsonify({"objects": result})
