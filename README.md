# SAHI with Your Own YOLOX

This repository contains the implementation of the SAHI using YOLOX.  
You can use your **own weights** and inference using SAHI.

```python
from sahi_predictor.yolox_sahi import YoloXDetectionModel
from yolox_model.predictor import YoloxPredictor
from const.values import COCO_LABELS

# initialize your own YOLOX model
yolox = YoloxPredictor(
    model_path="yolox_s.pth",
    depth=0.33,
    width=0.50,
    confthre=0.3,
    nmsthre=0.30,
    cls_names=COCO_LABELS,  # change this to your own class names
    input_size=(1024, 1024),
)

# inference using YOLOX
# outputs, numpy_img = yolox.inference(img="cars.jpg")


# initialize YOLOX-sahi model
yolox_sahi_model: YoloXDetectionModel = YoloXDetectionModel(
        model=yolox,
    )

# inference using SAHI
result = get_sliced_prediction(
    image=cv2.imread("cars.jpg"),
    detection_model=yolox_sahi_model,
    slice_height=1024,
    slice_width=1024,
    perform_standard_pred=False,
    overlap_height_ratio=0.2,
    overlap_width_ratio=0.2,
)
```

refer to the [notebooks/tutorials.ipynb](./notebooks/tutorials.ipynb) for more details.

## 1. install

### 1.1 clone repository

clone this repository with submodules as this repository includes yolox as a submodule.

```bash
git clone --recurse-submodules git@github.com:Nao-Y1996/yolox_sahi.git
```

If you cloned without `--recurse-submodules`, please run the following commands to get the yolo submodule.

```bash
git submodule init 
git submodule update
```

### 1.2 install libraries

```bash
poetry shell
poetry install
```

### 1.3. Install YOLOX

Before installing YOLOX, modify `yolox/requirements.txt` because dependencies in yolox is broken now(2024/10).

- Before modification
    ```requirements.txt
    onnx-simplifier==0.4.10
    ```
- After modification
    ```requirements.txt
    onnx-simplifier>=0.4.10
    ```
Then, install YOLOX as follows.

```bash
cd yolox
python setup.py develop
```
After installation, please revert the requirements.txt to the original state.

