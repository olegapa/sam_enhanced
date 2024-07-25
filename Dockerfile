FROM python:3.9

WORKDIR /segmentation

COPY . /segmentation

RUN pip install torch torchvision
RUN pip install opencv-python ftfy regex tqdm ttach tensorboard lxml cython
RUN pip install segmentation_models_pytorch
RUN pip install -U albumentations
RUN pip install pandas
WORKDIR /segmentation/CLIP-ES/pydensecrf
RUN python setup.py install

WORKDIR /segmentation/segment-anything
RUN pip install -e .
RUN pip install opencv-python pycocotools matplotlib onnxruntime onnx

RUN apt-get update
RUN apt-get --yes install libgl1
WORKDIR /segmentation
# Точка входа для контейнера
ENTRYPOINT ["python", "entrypoint.py"]