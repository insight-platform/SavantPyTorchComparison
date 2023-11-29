import time
import argparse
import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default='/workspace/data/deepstream_sample_720p.mp4')
    parser.add_argument('--model_path', type=str, default='/workspace/models/yolov8m.pt')
    return parser.parse_args()


def main(args):
    model = YOLO(args.model_path)
    cap = cv2.VideoCapture(args.file_path)
    num_frames = 0
    t0 = time.monotonic()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        _ = model(frame, half=True, device='0')

        num_frames += 1

    elapsed = time.monotonic() - t0
    cap.release()

    fps = num_frames / elapsed
    print(f" - Processed {num_frames} frames in {elapsed:.2f} seconds. ({fps:.2f} fps)")

if __name__ == '__main__':
    main(parse_args())
