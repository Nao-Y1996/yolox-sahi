
from typing import Any, List, Optional

import cv2
import numpy as np
import numpy.typing as npt
from sahi.annotation import BoundingBox
from sahi.models.base import DetectionModel as BaseDetectionModel
from sahi.predict import PredictionResult, get_sliced_prediction
from sahi.prediction import ObjectPrediction
from torch import Tensor

from yolox_model.predictor import YoloxPredictor


class YoloXDetectionModel(BaseDetectionModel):

    def __init__(
            self,
            model: YoloxPredictor,
            confidence_threshold: float = 0.5,
            nms_threshold=0.4,
    ):
        """
        Init object detection/instance segmentation model.

        Parameters
        ----------
        model: YoloxPredictor
            YoloxPredictor object
        confidence_threshold: float
            All predictions with score < confidence_threshold will be discarded. Defaults to 0.5.
        nms_threshold: float
            Non-maximum suppression threshold. Defaults to 0.4.

        """
        category_mapping = {}
        for i, label in enumerate(model.cls_names):
            category_mapping[str(i)] = label

        super().__init__(
            model=model,
            confidence_threshold=confidence_threshold,
            category_mapping=category_mapping,
            image_size=model.input_size[0],
        )
        self.classes = model.cls_names
        self.nms_threshold = nms_threshold

    def load_model(self) -> None:
        pass

    def set_model(self, model: Any, **kwargs) -> None:
        self.model: YoloxPredictor = model
        pass

    def perform_inference(self, img: str | npt.NDArray[np.uint8]) -> Tensor:
        """
        Perform inference on the given image.

        Parameters
        ----------
        img: str | npt.NDArray[np.uint8]
            Location of image or numpy image matrix to perform inference

        Returns
        -------
        torch.Tensor
            Predictions in the format [[x0, y0, x1, y1, score, score, cls_id], ...] where coordinates are in xyxy format and divided by image size.
        """
        if isinstance(img, str):
            image: npt.NDArray[np.uint8] = cv2.imread(img)
        else:
            image = img.copy()
        outputs, _ = self.model.inference(image)

        self._original_predictions = outputs[
            0]  # [[x0, y0, x1, y1, score, cls_id], ...]  coordinates are in xyxy format and divided by image size
        return outputs

    def _create_object_prediction_list_from_original_predictions(
            self,
            shift_amount_list: Optional[List[List[int]]] = None,
            full_shape_list: Optional[List[List[int]]] = None,
    ):
        if shift_amount_list is None:
            shift_amount_list = [[0, 0]]
        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        original_predictions = self._original_predictions
        bboxes = []
        class_ids = []
        scores = []

        if original_predictions is not None:
            bboxes = original_predictions[:, 0:4]
            class_ids = original_predictions[:, 6]
            scores = original_predictions[:, 4] * original_predictions[:, 5]

        shift_amount: List[int] = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        object_prediction_list = []
        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(class_ids[i])
            score = scores[i]
            if score < self.confidence_threshold:
                continue

            x0 = int(box[0])
            y0 = int(box[1])
            x1 = int(box[2])
            y1 = int(box[3])

            bbox = [x0, y0, x1, y1]

            object_prediction = ObjectPrediction(
                bbox=bbox,
                category_id=cls_id,
                category_name=self.category_mapping[str(cls_id)],
                shift_amount=shift_amount,
                score=score,
                full_shape=full_shape,
            )
            object_prediction_list.append(object_prediction)

        object_prediction_list_per_image = [object_prediction_list]
        self._object_prediction_list_per_image = object_prediction_list_per_image
        return object_prediction_list_per_image


def perform_sahi(
        image: str | npt.NDArray[np.uint8],
        model: YoloXDetectionModel,
        slice_height: int = 1024,
        slice_width: int = 1024,
        overlap_height_ratio: float = 0.3,
        overlap_width_ratio: float = 0.3,
        verbose: int = 1,
) -> tuple[list[list[float]], dict[str, float], npt.NDArray[np.uint8]]:
    """
    Perform inference on the given image and return the result of geometric tolerance detection and ocr inference.

    Parameters
    ----------
    image: npt.NDArray[Any]
        Location of image or numpy image matrix to slice
    model: YoloXDetectionModel
    slice_height: int
        Height of each slice.  Defaults to ``None``.
    slice_width: int
        Width of each slice.  Defaults to ``None``.
    overlap_height_ratio: float
        Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window
        of size 512 yields an overlap of 102 pixels).
        Default to ``0.3``.
    overlap_width_ratio: float
        Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window
        of size 512 yields an overlap of 102 pixels).
        Default to ``0.3``.
    verbose: int
        0: no print
        1: print number of slices (default)
        2: print number of slices and slice/prediction durations

    Returns
    -------
    tuple[list[list[float]], dict[str, float], npt.NDArray[np.uint8]]
        Object detection results, durations in seconds, and image with bounding boxes.
    """

    if isinstance(image, str):
        image: npt.NDArray[np.uint8] = cv2.imread(image)
    else:
        image = image.copy()
    print(type(image))

    result: PredictionResult = get_sliced_prediction(
        image,
        detection_model=model,
        slice_height=slice_height,
        slice_width=slice_width,
        perform_standard_pred=False,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
        verbose=verbose,
    )

    durations_in_seconds = result.durations_in_seconds  # {'slice': XXs, 'prediction': XXs}
    numpy_img: np.ndarray = np.array(result.image)

    object_predictions: list[ObjectPrediction] = result.object_prediction_list
    outputs = []
    for object_prediction in object_predictions:
        bbox: BoundingBox = object_prediction.bbox
        x1, y1, x2, y2 = map(int, [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy])
        score = object_prediction.score.value.item()
        output = [x1, y1, x2, y2, score, object_prediction.category.id]
        outputs.append(output)
    return outputs, durations_in_seconds, numpy_img
