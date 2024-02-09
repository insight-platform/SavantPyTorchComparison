# SavantPyTorchComparison

This project aims to demonstrate a few alternative ways to utilize a Pytorch detection model and compare their performance. To this end, three equivalent pipelines were implemented:

1. Pytorch pipeline that receives its input from OpenCV VideoCapture in a Numpy array (host memory);
1. Pytorch pipeline that receives its input from Torchaudio StreamReader with hardware-accelerated video decoder in a GPU Torch tensor (device memory);
1. [Savant](https://github.com/insight-platform/Savant) pipeline, based on NVIDIA Deepstream+TensorRT.

Common pipeline inference parameters:

- GPU inference
- 640x640 inference dimensions
- 1 batch size
- fp16 mode

## Prerequisites

### Docker images

Benchmark pipelines are run in Docker containers.

Build the Pytorch container by running:

```bash
make build-pytorch
```

Pull the Savant container by running:

```bash
make pull-savant
```

### Input video

Benchmark pipelines use an h264 video as input. Download it by running

```bash
make get-test-video
```

Check that `data/deepstream_sample_720p.mp4` file exists.

### Models

Pytorch pipelines use `YOLOv8m` model from [ultralytics](https://github.com/ultralytics/ultralytics). Download the weights by running:

```bash
make get-pytorch-model
```

Check that `pytorch_weights/yolov8m.pt` file exists.

Savant pipeline uses the same model exported to ONNX format. Run the export with:

```bash
make run-export-onnx
```

Check that `cache/models/yolov8m_pipeline/yolov8m/yolov8m.onnx` file exists.

## Run

Run the OpenCV VideoCapture version of the pipeline with:

```bash
make run-pytorch-opencv
```

Run the Torchaudio + HW decoder version of the pipeline with:

```bash
make run-pytorch-hw-decode
```

Run the Savant version of the pipeline with:

```bash
make run-savant
```

## Results

Test              | FPS
------------------|----
Pytorch OpenCV    | 75
Pytorch HW Decode | 107
Savant            | 255

### Hardware

Hardware used:

| GPU              | CPU                               | RAM, Gi |
|------------------|-----------------------------------|---------|
| GeForce RTX 2080 | Intel Core i5-8600K CPU @ 3.60GHz | 31      |
