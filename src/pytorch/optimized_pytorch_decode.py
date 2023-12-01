import cv2
import torch
from torchaudio.io import StreamReader
from ultralytics.utils.ops import non_max_suppression, scale_boxes
from utils import FPSTimer, get_arg_parser, setup_raw_pytorch_model


def yuv_to_rgb(frames: torch.tensor) -> torch.tensor:
    """Converts YUV BCHW dims torch tensor to RGB BCHW dims torch tensor

    :param frames: YUV BCHW dims torch tensor
    :return: RGB BCHW dims torch tensor
    """
    frames = frames.to(torch.float)
    frames /= 255
    y = frames[..., 0, :, :]
    u = frames[..., 1, :, :] - 0.5
    v = frames[..., 2, :, :] - 0.5

    r = y + 1.14 * v
    g = y + -0.396 * u - 0.581 * v
    b = y + 2.029 * u

    rgb = torch.stack([r, g, b], 1)
    rgb = rgb.clamp(0, 1)
    return rgb


def frame_to_np(tensor):
    frame = (tensor * 255).clamp(0, 255).to(torch.uint8)
    frame = frame.numpy(force=True).transpose(1, 2, 0)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame


def get_video_size(file_path):
    stream_reader = StreamReader(file_path)
    stream_reader.add_video_stream(
        1,
        decoder='h264_cuvid',
        hw_accel='cuda:0',
    )
    stream_reader.fill_buffer()
    (video,) = stream_reader.pop_chunks()
    b, c, h, w = video.shape
    return h, w


def get_decoder_option(infer_size, video_size):
    """Get HW decoder options that produce infer_size frames from video
    while not introducing any distortion to the video contents.
    The HW decoder cannot pad the frames, so aspect ratio changes
    must be done by cropping the video.
    HW decoder order of operations is crop -> resize.

    :param infer_size: (h, w)
    :param video_size: (h, w)
    :return: dict with crop+resize decoder options
    """
    video_height, video_width = video_size
    infer_height, infer_width = infer_size

    old_aspect = video_width / video_height
    new_aspect = infer_width / infer_height

    if old_aspect > new_aspect:
        # Crop width
        crop_width = int(video_height * new_aspect)
        crop_height = video_height
        crop_left = (video_width - crop_width) // 2
        crop_right = video_width - crop_width - crop_left
        crop_top = 0
        crop_bottom = 0
    else:
        # Crop height
        crop_width = video_width
        crop_height = int(video_width / new_aspect)
        crop_left = 0
        crop_right = 0
        crop_top = (video_height - crop_height) // 2
        crop_bottom = video_height - crop_height - crop_top

    return {
        'crop': f'{crop_top}x{crop_bottom}x{crop_left}x{crop_right}',
        'resize': f'{infer_width}x{infer_height}',
    }


def main(args):
    torch_model = setup_raw_pytorch_model(args.model_path)

    infer_shape = args.infer_height, args.infer_width
    orig_shape = get_video_size(args.file_path)

    stream_reader = StreamReader(args.file_path)
    decoder_option = get_decoder_option(infer_shape, orig_shape)
    stream_reader.add_video_stream(
        args.batch_size,
        decoder='h264_cuvid',
        hw_accel='cuda:0',
        decoder_option=decoder_option,
    )

    # Start the main loop
    with FPSTimer() as timer:
        for (stream_chunk,) in stream_reader.stream():
            timer.num_frames += stream_chunk.shape[0]
            # Preprocess
            batch_frames = yuv_to_rgb(stream_chunk)
            batch_frames = batch_frames.half()

            with torch.inference_mode():
                # Inference
                batch_preds = torch_model(batch_frames)

                # Postprocess
                batch_boxes = non_max_suppression(batch_preds)
                for i, boxes in enumerate(batch_boxes):
                    # Optionally, visually check the results
                    # Drawing is intentionally not included in the comparison benchmark
                    # Visualization has to be done before scaling the boxes for the HW decoder
                    # because the HW decoder does not produce original size frames
                    # and the scale_boxes op modifies the boxes in-place
                    # frame = frame_to_np(batch_frames[i])
                    # for box in boxes:
                    #     pt1 = (int(box[0]), int(box[1]))
                    #     pt2 = (int(box[2]), int(box[3]))
                    #     cv2.rectangle(frame, pt1, pt2, (0, 255, 0))
                    # cv2.imwrite('test.jpg', frame)

                    boxes[:, :4] = scale_boxes(infer_shape, boxes[:, :4], orig_shape)
                    # Move to cpu
                    boxes = boxes.numpy(force=True)


if __name__ == '__main__':
    parser = get_arg_parser()
    main(parser.parse_args())
