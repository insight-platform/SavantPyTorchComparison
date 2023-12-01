PYTORCH_IMAGE_NAME := savant-pt-compare-pytorch
SAVANT_IMAGE_NAME := savant-pt-compare-savant
SAVANT_MODULE_NAME := yolov8_pipeline

build-savant:
	docker build -t $(SAVANT_IMAGE_NAME) docker/savant

build-pytorch:
	docker build -t $(PYTORCH_IMAGE_NAME) docker/pytorch

run-savant:
	docker run --rm --gpus=all \
		-e DOWNLOAD_PATH=/cache/downloads/$(SAVANT_MODULE_NAME) \
		-e MODEL_PATH=/cache/models/$(SAVANT_MODULE_NAME) \
		-e CUPY_CACHE_DIR=/cache/cupy \
		-e NUMBA_CACHE_DIR=/cache/numba \
		-e GST_DEBUG \
		-e LOGLEVEL \
		-e METRICS_FRAME_PERIOD \
		-e GST_DEBUG_COLOR_MODE=off \
		-v $(shell pwd)/src/savant:/opt/savant/samples/$(SAVANT_MODULE_NAME) \
		-v $(shell pwd)/data:/data:ro \
		-v $(shell pwd)/cache:/cache \
		$(SAVANT_IMAGE_NAME) samples/$(SAVANT_MODULE_NAME)/module_perf.yml

run-pytorch-opencv:
	docker run --rm --gpus=all \
		-v $(shell pwd)/src/pytorch:/workspace/src \
		-v $(shell pwd)/data:/workspace/data:ro \
		-v $(shell pwd)/pytorch_models:/workspace/models \
		$(PYTORCH_IMAGE_NAME) src/naive_opencv_pytorch.py

run-pytorch-hw-decode:
	docker run --rm --gpus=all \
		-v $(shell pwd)/src/pytorch:/workspace/src \
		-v $(shell pwd)/data:/workspace/data:ro \
		-v $(shell pwd)/pytorch_models:/workspace/models \
		$(PYTORCH_IMAGE_NAME) src/optimized_pytorch_decode.py
