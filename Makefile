PYTORCH_IMAGE_NAME := savant-pt-compare-pytorch
SAVANT_IMAGE_NAME := ghcr.io/insight-platform/savant-deepstream:latest
SAVANT_MODULE_NAME := yolov8_pipeline

get-test-video:
	mkdir -p data
	curl -o data/deepstream_sample_720p.mp4 \
	https://eu-central-1.linodeobjects.com/savant-data/demo/deepstream_sample_720p.mp4

get-pytorch-model:
	mkdir -p pytorch_weights
	wget --output-document pytorch_weights/yolov8m.pt \
	https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt

build-pytorch:
	docker build -t $(PYTORCH_IMAGE_NAME) docker/pytorch

pull-savant:
	docker pull $(SAVANT_IMAGE_NAME)

run-savant:
	docker run --rm --gpus=all \
		-e MODEL_PATH=/cache/models/$(SAVANT_MODULE_NAME) \
		-v `pwd`/src/savant:/opt/savant/samples/$(SAVANT_MODULE_NAME) \
		-v `pwd`/data:/data:ro \
		-v `pwd`/cache:/cache \
		$(SAVANT_IMAGE_NAME) samples/$(SAVANT_MODULE_NAME)/module_perf.yml

run-export-onnx:
	docker run --rm --gpus=all \
		-v `pwd`/src/pytorch:/workspace/src \
		-v `pwd`/data:/workspace/data:ro \
		-v `pwd`/pytorch_weights:/workspace/models \
		-v `pwd`/cache:/cache \
		$(PYTORCH_IMAGE_NAME) src/yolo_onnx_export.py

run-pytorch-opencv:
	docker run --rm --gpus=all \
		-v `pwd`/src/pytorch:/workspace/src \
		-v `pwd`/data:/workspace/data:ro \
		-v `pwd`/pytorch_weights:/workspace/models \
		$(PYTORCH_IMAGE_NAME) src/naive_opencv_pytorch.py

run-pytorch-hw-decode:
	docker run --rm --gpus=all \
		-v `pwd`/src/pytorch:/workspace/src \
		-v `pwd`/data:/workspace/data:ro \
		-v `pwd`/pytorch_weights:/workspace/models \
		$(PYTORCH_IMAGE_NAME) src/optimized_pytorch_decode.py
