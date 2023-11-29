import cv2
import torch
def draw_results(results, frame):
    for result in results:
        xyxy = result.boxes.xyxy.numpy(force=True)
        print(xyxy)
        for box in xyxy:
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            cv2.rectangle(frame, pt1, pt2, (0, 255, 0))

def yuv_to_rgb(frames):
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