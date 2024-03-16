FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY ./detector_app /app
WORKDIR /app

RUN pip install -r requirements.txt

ENTRYPOINT ["python3", "detection_server.py"]