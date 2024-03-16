import grpc
from concurrent import futures

import inference_pb2_grpc, inference_pb2
from inference_utils import configure_inference_setup, infer


class InstanceDetector(inference_pb2_grpc.InstanceDetectorServicer):
    def __init__(self):
        model, transform, labels = configure_inference_setup()
        self.model = model
        self.transform = transform
        self.labels = labels

    def Predict(self, request, context):
        results = infer(request.url, model=self.model, transform=self.transform, labels=self.labels)
        return inference_pb2.InstanceDetectorOutput(objects=results)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    inference_pb2_grpc.add_InstanceDetectorServicer_to_server(InstanceDetector(), server)
    server.add_insecure_port('[::]:9090')
    server.start()
    server.wait_for_termination()
