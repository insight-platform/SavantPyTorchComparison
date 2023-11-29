import time
import argparse
from torchaudio.io import StreamReader
from ultralytics import YOLO
from utils import yuv_to_rgb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/workspace/data/deepstream_sample_720p.mp4')
    parser.add_argument('--model_path', type=str, default='/workspace/models/yolov8m.pt')
    parser.add_argument('--batch_size', type=int, default=4)
    return parser.parse_args()


def main(args):
    model = YOLO(args.model_path)
    s = StreamReader(args.file_path)
    decoder_option={"resize": "640x384"}
    s.add_video_stream(args.batch_size, decoder="h264_cuvid", hw_accel="cuda:0", decoder_option=decoder_option)

    num_frames = 0
    t0 = time.monotonic()
    for (frames,) in s.stream():
        num_frames += frames.shape[0]

        frames = yuv_to_rgb(frames)

        _ = model(frames, half=True, device='0')

    elapsed = time.monotonic() - t0
    fps = num_frames / elapsed

    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")


if __name__ == '__main__':
    main(parse_args())
