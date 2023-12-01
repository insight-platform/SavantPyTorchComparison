import argparse
import time

import numpy as np
import torch
from ultralytics import YOLO


def get_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--file_path', type=str, default='/workspace/data/deepstream_sample_720p.mp4'
    )
    parser.add_argument(
        '--model_path', type=str, default='/workspace/models/yolov8m.pt'
    )

    parser.add_argument(
        '--infer_height',
        type=int,
        default=640,
        help='Inference height, must be multiple of 32.',
    )
    parser.add_argument(
        '--infer_width',
        type=int,
        default=640,
        help='Inference width, must be multiple of 32.',
    )
    parser.add_argument(
        '--batch_size', type=int, default=4, help='Inference batch size.'
    )
    return parser


def setup_raw_pytorch_model(weights_path) -> torch.nn.Module:
    model = YOLO(weights_path, task='detect')
    dummy_input = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    _ = model(dummy_input, half=True, device='0', verbose=False)

    torch_model = model.predictor.model
    assert isinstance(torch_model, torch.nn.Module)
    assert not torch_model.training
    assert next(torch_model.parameters()).is_cuda
    assert next(torch_model.parameters()).dtype == torch.float16
    return torch_model


class FPSTimer:
    def __init__(self):
        self.num_frames = 0
        self.start = time.monotonic()

    def __enter__(self):
        self.num_frames = 0
        self.start = time.monotonic()
        return self

    def __exit__(self, *args):
        elapsed = time.monotonic() - self.start
        fps = self.num_frames / elapsed
        print(
            f' - Processed {self.num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)'
        )
