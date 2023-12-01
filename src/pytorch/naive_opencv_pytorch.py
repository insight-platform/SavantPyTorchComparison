import os

import cv2
import numpy as np
import torch
from resize_util import resize_preserving_aspect
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from utils import FPSTimer, get_arg_parser, setup_raw_pytorch_model


def main(args):
    # Load YOLOv8 model
    torch_model = setup_raw_pytorch_model(args.model_path)

    # OpenCV video capture with hardware acceleration explicitly disabled
    cap = cv2.VideoCapture(
        args.file_path,
        cv2.CAP_ANY,
        (cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE),
    )

    # Init resize transform
    infer_shape = args.infer_height, args.infer_width
    orig_shape = (
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
    )

    print(
        'Starting inference loop. Parameters:\n'
        f'Model: {os.path.basename(args.model_path)}\n'
        f'Original resolution HxW {orig_shape} -> infer resolution HxW {infer_shape}\n'
        f'Infer batch size: {args.batch_size}\n'
        f'OpenCV cap backend: {cap.getBackendName()}, no HW acceleration.'
    )
    # Start the main loop
    with FPSTimer() as timer:
        while cap.isOpened():
            orig_frames = []
            for _ in range(args.batch_size):
                read_success, frame = cap.read()
                if not read_success:
                    print(
                        "Can't receive frame (stream end?). Stopping new frames reading."
                    )
                    break
                orig_frames.append(frame)
            timer.num_frames += len(orig_frames)

            if not orig_frames:
                break

            # Preprocess
            # Letterbox resize
            batch_frames = [
                resize_preserving_aspect(frame, infer_shape) for frame in orig_frames
            ]
            batch_frames = np.stack(batch_frames, axis=0)
            # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            batch_frames = batch_frames[..., ::-1].transpose(0, 3, 1, 2)
            batch_frames = np.ascontiguousarray(batch_frames)
            batch_frames = torch.from_numpy(batch_frames)
            batch_frames = batch_frames.cuda()
            batch_frames = batch_frames.half()
            batch_frames /= 255.0

            with torch.inference_mode():
                # Inference
                batch_preds = torch_model(batch_frames)

                # Postprocess
                batch_boxes = non_max_suppression(batch_preds)
                for i, boxes in enumerate(batch_boxes):
                    boxes[:, :4] = scale_boxes(infer_shape, boxes[:, :4], orig_shape)
                    # Move to cpu
                    boxes = boxes.numpy(force=True)

                    # Optionally, visually check the results
                    # Drawing is intentionally not included in the comparison benchmark
                    # frame = orig_frames[i]
                    # for box in boxes:
                    #     pt1 = (int(box[0]), int(box[1]))
                    #     pt2 = (int(box[2]), int(box[3]))
                    #     cv2.rectangle(frame, pt1, pt2, (0, 255, 0))
                    # cv2.imwrite('/workspace/src/test.jpg', frame)

            if not read_success:
                break

    cap.release()


if __name__ == '__main__':
    parser = get_arg_parser()
    main(parser.parse_args())
