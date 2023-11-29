SAVANT_IMAGE_NAME := savant-compare-module
SAVANT_MODULE_NAME := yolov8_pipeline

build-savant:
	docker build -t $(SAVANT_IMAGE_NAME) docker_savant

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
		-v $(shell pwd)/savant_module:/opt/savant/samples/$(SAVANT_MODULE_NAME) \
		-v $(shell pwd)/data:/data:ro \
		-v $(shell pwd)/cache:/cache \
		$(SAVANT_IMAGE_NAME) samples/$(SAVANT_MODULE_NAME)/module_perf.yml
