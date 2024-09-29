"""
Mask R-CNN
Train on the solar panel dataset and detect solar panel.

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 solar_panel.py train --dataset=/path/to/solar_panel/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 solar_panel.py train --dataset=/path/to/solar_panel/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 solar_panel.py train --dataset=/path/to/solar_panel/dataset --weights=imagenet

    # Detect solar panel in an image and display it
    python3 solar_panel.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>
"""

import os
import sys
import json
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class SolarPanelConfig(Config):
    """
    Configuration for training on the solar panel dataset.
    Derives from the base Config class and overrides some values.

    """

    # Give the configuration a recognizable name
    NAME = "solar-panel"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + solar panel

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################


class SolarPanelDataset(utils.Dataset):

    def load_solar_panel(self, dataset_dir, subset):
        """
        Load a subset of the solar panel dataset.

        Arguments:
            dataset_dir: Root directory of the dataset.
            subset: Subset to load: train or val

        """
        # Add classes. We have only one class to add.
        self.add_class("solar-panel", 1, "solar-panel")

        # Train or validation dataset?
        assert subset in ["train", "val"]
        annotations_dir = os.path.join(dataset_dir, subset)

        # Load annotations from json
        # This process is modified since current json format of
        # annotations is COCO, not VGG.
        with open(os.path.join(annotations_dir, "annotations.json")) as fp:
            json_data = json.load(fp)

        # Initialize images and annotations
        images = {image["id"]: image for image in json_data["images"]}
        annotations = {}

        # Create annotations into dictionary.
        for annotation in json_data["annotations"]:
            if (image_id := annotation["image_id"]) not in annotations:
                annotations[image_id] = []

            annotations[image_id].append(annotation)

        # Print load result
        print(f"loaded {len(images)} images.")
        print(f"loaded {len(annotations)} annotations.")

        # Add images
        for image_id, annotation in annotations.items():
            polygons = [
                {
                    "all_points_x": list(
                        map(lambda x: x - 0.01, a["segmentation"][0][0::2])
                    ),
                    "all_points_y": list(
                        map(lambda x: x - 0.01, a["segmentation"][0][1::2])
                    ),
                    "name": "polygon",
                }
                for a in annotation
            ]

            image_path = os.path.join(
                dataset_dir,
                "images",
                images[image_id]["file_name"],
            )
            height = images[image_id]["height"]
            width = images[image_id]["width"]

            self.add_image(
                "solar-panel",
                image_id=images[image_id]["file_name"],
                path=image_path,
                height=height,
                width=width,
                polygons=polygons,
            )

    def load_mask(self, image_id):
        """
        Generate instance masks for an image.
        
        Returns:
            masks: A bool array of shape [height, width, instance count] with one mask per instance.
            class_ids: a 1D array of class IDs of the instance masks.

        """
        assert self.image_info[image_id]["source"] == "solar-panel"

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros(
            [info["height"], info["width"], len(info["polygons"])], dtype=np.uint8
        )

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "solar-panel":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """
    Train the model.

    """
    # Training dataset.
    dataset_train = SolarPanelDataset()
    dataset_train.load_solar_panel(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = SolarPanelDataset()
    dataset_val.load_solar_panel(args.dataset, "val")
    dataset_val.prepare()

    print("Training network heads")

    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=100,
        layers="heads",
    )


def detect(model, image_path=None):
    """
    Detect solar panel from the given image.

    """
    if image_path:
        # Run model detection and display result
        print(f"Running on {args.image}")

        # Read image
        image = skimage.io.imread(args.image)

        # Use only first three color channel
        if image.shape[-1] == 4:
            image = image[..., :3]

        # Detect objects
        r = model.detect([image], verbose=1)[0]

        # Visualize the detected objects.
        import mrcnn.visualize

        mrcnn.visualize.display_instances(
            image=image,
            boxes=r["rois"],
            masks=r["masks"],
            class_ids=r["class_ids"],
            class_names=["BG", "solar-panel"],
            scores=r["scores"],
        )


############################################################
#  Training
############################################################

if __name__ == "__main__":
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train Mask R-CNN to detect solar panels."
    )
    parser.add_argument(
        "command",
        metavar="<command>",
        help="'train' or 'detect'",
    )
    parser.add_argument(
        "--dataset",
        required=False,
        metavar="/path/to/solar_panel/datasets/",
        help="Directory of the solar panel datasets",
    )
    parser.add_argument(
        "--weights",
        required=True,
        metavar="/path/to/weights.h5",
        help="Path to weights .h5 file or 'coco'",
    )
    parser.add_argument(
        "--logs",
        required=False,
        default=DEFAULT_LOGS_DIR,
        metavar="/path/to/logs/",
        help="Logs and checkpoints directory (default=logs/)",
    )
    parser.add_argument(
        "--image",
        required=False,
        metavar="path or URL to image",
        help="Image to detect solar panel",
    )
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    elif args.command == "detect":
        assert args.image, "Provide --image to apply detection"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = SolarPanelConfig()

    else:

        class InferenceConfig(SolarPanelConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(
            mode="training",
            config=config,
            model_dir=args.logs,
        )

    else:
        model = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=args.logs,
        )

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH

        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)

    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()

    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()

    else:
        weights_path = args.weights

    # Load weights
    print(f"Loading weights from {weights_path}")

    if args.weights.lower() == "coco":
        # Exclude the last layers
        # because they require a matching number of classes
        model.load_weights(
            weights_path,
            by_name=True,
            exclude=[
                "mrcnn_class_logits",
                "mrcnn_bbox_fc",
                "mrcnn_bbox",
                "mrcnn_mask",
            ],
        )

    else:
        model.load_weights(weights_path, by_name=True)

    # Train or detect
    if args.command == "train":
        train(model)

    elif args.command == "detect":
        detect(model, image_path=args.image)

    else:
        print(f"'{args.command}' is not recognized. Use 'train' or 'detect'")
