# Solar Panel Detection

This is an solar panel detection model using mask-rcnn.

## Installation

1. Clone this repository

   ```bash
   git clone https://github.com/kaist-220v/mask-rcnn.git maskrcnn
   ```

2. Create environment with anaconda and install dependencies:

   ```bash
   conda env create -f environment.yml
   ```

3. Download pre-trained COCO weights (mask_rcnn_coco.h5) from the [releases page](https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5).

4. (optional) If you want to use GPU for detect/train, install these drivers:
   - NVIDIA Driver version `452.39` at least
   - CUDA Toolkit version `11.8`
   - cuDNN Library version `8.7.0`
   - install `tensorflow[and-cuda]` using pip

5. If you want to train this model, import dataset and create directory as follow:

```text
maskrcnn/
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

## Use `solar_panel.py` for solar panel detection

### Detect Solar Panel Using the Provided Weights

Detect solar panel on an image:

```bash
python solar_panel.py detect --weights=/path/to/mask_rcnn/mask_rcnn_solar_panel.h5 --image=<file name or URL>
```

### Train the Solar Panel Model

Train a new model starting from pre-trained COCO weights

```bash
python solar_panel.py train --dataset=/path/to/solar_panel/dataset --weights=coco
```

Resume training a model that you had trained earlier

```bash
python solar_panel.py train --dataset=/path/to/solar_panel/dataset --weights=last
```

Train a new model starting from ImageNet weights

```bash
python solar_panel.py train --dataset=/path/to/solar_panel/dataset --weights=imagenet
```

The code in `solar_panel.py` is set to train for 10K steps (100 epochs of 100 steps each), and using a batch size of 2. Update the schedule to fit your needs.

### Test the model accuracy (mAP@IoU-0.5)

```bash
python solar_panel.py test --dataset=/path/to/solar_panel/dataset --weights=/path/to/weights/file.h5
```
