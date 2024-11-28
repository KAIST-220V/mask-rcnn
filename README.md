# Solar Panel Detection

This is an solar panel detection model, which has been modified from balloon splach sample.

## Installation

1. Set environment by following [this instruction](https://github.com/KAIST-220V/Mask-RCNN_TF2.14.0/blob/main/README.md).
2. (optional) If you want to use GPU for detect/train, install these drivers:
   - NVIDIA Driver version `452.39` at least
   - CUDA Toolkit version `11.8`
   - cuDNN Library version `8.7.0`
   - install `tensorflow[and-cuda]` using pip
3. If you want to train this model, import dataset and create directory as follow:

```
solar_panel/
├── datasets/
│   ├── images/
│   │   └── ...
│   ├── train/
│   │   └── annotations.json
│   └── val/
│       └── annotations.json
├── README.md
└── solar_panel.py
```

Now you are all set!

## Detect Solar Panel Using the Provided Weights

Detect solar panel on an image:

```
python3 solar_panel.py detect --weights=/path/to/mask_rcnn/mask_rcnn_solar_panel.h5 --image=<file name or URL>
```

## Train the Solar Panel Model

Train a new model starting from pre-trained COCO weights

```
python3 solar_panel.py train --dataset=/path/to/balloon/dataset --weights=coco
```

Resume training a model that you had trained earlier

```
python3 solar_panel.py train --dataset=/path/to/balloon/dataset --weights=last
```

Train a new model starting from ImageNet weights

```
python3 solar_panel.py train --dataset=/path/to/balloon/dataset --weights=imagenet
```

The code in `solar_panel.py` is set to train for 10K steps (100 epochs of 100 steps each), and using a batch size of 2. Update the schedule to fit your needs.

## Test the model accuracy (mAP @IoU-0.5)

```
python3 solar_panel.py test --dataset=/path/to/solar_panel/dataset --weights=/path/to/weights/file.h5
```
