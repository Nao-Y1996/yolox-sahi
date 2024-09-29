
import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from yolox.data.data_augment import ValTransform
from yolox.models import YOLOPAFPN, YOLOX, YOLOXHead
from yolox.utils import get_local_rank, postprocess


def load_model_by_params(
        model_path: str, depth: float, width: float, num_classes: int, act: str = "silu"
) -> nn.Module:
    def get_model():
        """
        Get the YOLOX model.
        This function is copied from get_model() in yolox.exp.yolox_base.py
        """

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [256, 512, 1024]
        backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
        head = YOLOXHead(num_classes, width, in_channels=in_channels, act=act)
        model = YOLOX(backbone, head)

        model.apply(init_yolo)
        model.head.initialize_biases(1e-2)
        model.eval()
        return model

    model = get_model()
    rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        loc = "cuda:{}".format(rank)
        model.cuda(rank)
    else:
        loc = "cpu"
    ckpt = torch.load(model_path, map_location=loc)
    model.load_state_dict(ckpt["model"])
    return model


class YoloxPredictor(object):
    """
    This class is a customized version of Predictor in yolox/tools/demo.py to initialize the model with the given parameters.
    """

    def __init__(
            self,
            model_path: str,
            depth: float,
            width: float,
            cls_names: list[str],
            act: str = "silu",
            confthre: float = 0.3,
            nmsthre: float = 0.65,
            input_size: tuple[int, int] = (640, 640),
            device="cpu",
            fp16=False,
            legacy=False,
    ):
        self.cls_names = cls_names
        self.num_classes = len(cls_names)
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.input_size = input_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        self.model = load_model_by_params(model_path, depth, width, self.num_classes, act)

    def inference(self, img: str | npt.NDArray[np.uint8]) -> tuple[Tensor, npt.NDArray[np.uint8]]:
        """
        Perform inference on the given image.

        Parameters
        ----------
        img: str | npt.NDArray[np.uint8]
            Location of image or numpy image matrix to perform inference on.

        Returns
        -------
        tuple[Tensor, npt.NDArray[np.uint8]]
            Predictions and image with bounding boxes.
            Tensor: Predictions in the format [[x0, y0, x1, y1, score, score, cls_id], ...]
            npt.NDArray[np.uint8]: Original image. (same as input image)
        """

        if isinstance(img, str):
            image: npt.NDArray[np.uint8] = cv2.imread(img)
        else:
            image = img.copy()

        preproc_image, _ = self.preproc(image, None, self.input_size)
        image_tensor: Tensor = torch.from_numpy(preproc_image).unsqueeze(0)  # add a batch dimension
        image_tensor = image_tensor.float()
        if self.device == "gpu":
            image_tensor = image_tensor.cuda()
            if self.fp16:
                image_tensor = image_tensor.half()  # to FP16
        outputs = self.model(image_tensor)
        outputs = postprocess(
            outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=True
        )

        if outputs[0] is not None:
            prediction_result = outputs[0].cpu()
            bboxes = prediction_result[:, 0:4]
            # in preproc, image is resized by image_size * resize_ratio defined below. see yolox.data.data_augment.py
            resize_ratio = min(self.input_size[0] / image.shape[0], self.input_size[1] / image.shape[1])
            bboxes /= resize_ratio  # resize the bbox to the original image size
            prediction_result[:, 0:4] = bboxes
            outputs[0] = prediction_result
            # outputs: [[x1, y1, x2, y2, obj_conf, class_conf, cls_id], ...]
        return outputs, image

