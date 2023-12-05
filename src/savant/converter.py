"""YOLOv8 detector output to bbox converter."""
from typing import Tuple

import numpy as np

from savant.base.converter import BaseObjectModelOutputConverter
from savant.base.model import ObjectModel
from savant.utils.nms import nms_cpu


class TensorToBBoxConverter(BaseObjectModelOutputConverter):
    """YOLOv8 detector output to bbox converter.

    :param confidence_threshold: Select detections with confidence
        greater than specified.
    :param nms_iou_threshold: IoU threshold for NMS.
    :param top_k: Maximum number of output detections.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.25,
        nms_iou_threshold: float = 0.5,
        top_k: int = 300,
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.top_k = top_k
        super().__init__()

    def __call__(
        self,
        *output_layers: np.ndarray,
        model: ObjectModel,
        roi: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Converts detector output layer tensor to bbox tensor.

        Converter is suitable for PyTorch YOLOv8 models.
        Assumed one output layer with shape (84, 8400).
        Outputs best class only for each detection.

        :param output_layers: Output layer tensor list.
        :param model: Model definition, required parameters: input tensor shape,
            maintain_aspect_ratio
        :param roi: [top, left, width, height] of the rectangle
            on which the model infers
        :return: BBox tensor (class_id, confidence, xc, yc, width, height)
            offset by roi upper left and scaled by roi width and height
        """
        # unpack list + (84, 8400) -> (8400, 84)
        preds = np.transpose(output_layers[0])

        # confidence threshold filter applied to all classes
        keep = np.amax(preds[:, 4:], axis=1) > self.confidence_threshold
        if not keep.any():
            return np.float32([])
        preds = preds[keep]

        # pick highest confidence class for each detection
        class_ids = np.argmax(preds[:, 4:], axis=1, keepdims=True)
        confs = np.take_along_axis(preds[:, 4:], class_ids, axis=1).astype(np.float32)

        # offset coordinates boxes by per-class values
        # so that boxes of different classes do not intersect
        # and the nms can be performed per-class
        offset_boxes = preds[:, :4] + (class_ids * max(roi[2:])).astype(np.float32)
        keep = nms_cpu(
            offset_boxes,
            np.squeeze(confs),
            self.nms_iou_threshold,
            self.top_k,
        )
        if not keep.any():
            return np.float32([])

        class_ids = class_ids[keep].astype(np.float32)
        confs = confs[keep]
        xywh = preds[keep, :4]
        # roi width / model input width
        ratio_width = roi[2] / model.input.shape[2]
        # roi height / model input height
        ratio_height = roi[3] / model.input.shape[1]
        xywh *= max(ratio_width, ratio_height)
        xywh[:, 0] += roi[0]
        xywh[:, 1] += roi[1]
        bbox_output = np.concatenate((class_ids, confs, xywh), axis=1)
        return bbox_output
