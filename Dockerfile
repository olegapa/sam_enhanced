FROM python:3.10

WORKDIR /segmentation

COPY . /segmentation
RUN apt-get update && apt-get install -y docker.io
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

RUN apt-get update
RUN apt-get --yes install libgl1

WORKDIR /segmentation/CLIP-ES/pydensecrf
RUN python setup.py install

WORKDIR /segmentation/segment_anything
RUN pip install -e .
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx

WORKDIR /segmentation/sam2
RUN pip install -e .

WORKDIR /segmentation
# Точка входа для контейнера
ENTRYPOINT ["python", "new_entrypoint.py"]