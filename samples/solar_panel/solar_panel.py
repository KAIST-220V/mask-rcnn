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

    # Test a mAP accuracy of given model
    python3 solar_panel.py test --dataset=/path/to/solar_panel/dataset --weights=/path/to/weights/file.h5

    # Detect solar panel in an image and display it
    python3 solar_panel.py detect --weights=/path/to/weights/file.h5 --image=<URL or path to file>
"""

import os
import sys
import json
import numpy as np
import skimage.draw
import skimage.io
import skimage.util

# Root directory of the project
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# ignore decompression bomb attack warning
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

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
        assert subset in ["train", "val", "test"]
        annotations_dir = os.path.join(dataset_dir, subset)

        # Load annotations from json
        # This process is modified since current json format of
        # annotations is COCO, not VGG.
        with open(os.path.join(annotations_dir, "annotations.json")) as fp:
            json_data = json.load(fp)

        # Create images into dictionary
        images = {image["id"]: image for image in json_data["images"]}

        # Create annotations into dictionary
        from collections import defaultdict

        annotations = defaultdict(list)
        {
            annotations[annotation["image_id"]].append(annotation)
            for annotation in json_data["annotations"]
        }

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

        # Return mask, and array of class IDs of each instance.
        # Since we have one class ID only, we return an array of 1s
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
        epochs=30,
        layers="all",
    )


def test(model, limit=50):
    """
    Test the model.

    """
    from mrcnn.utils import extract_bboxes, compute_ap

    APs = []

    # Test dataset
    dataset_test = SolarPanelDataset()
    dataset_test.load_solar_panel(args.dataset, "test")
    dataset_test.prepare()

    # Loop through each image in the validation dataset
    for image_id in dataset_test.image_ids[:limit]:
        # Load the image and ground truth data
        image = dataset_test.load_image(image_id)
        gt_masks, gt_class_ids = dataset_test.load_mask(image_id)
        gt_boxes = extract_bboxes(gt_masks)

        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]

        # Compute AP
        AP, precisions, recalls, overlaps = compute_ap(
            gt_boxes,
            gt_class_ids,
            gt_masks,
            r["rois"],
            r["class_ids"],
            r["scores"],
            r["masks"],
        )
        APs.append(AP)

    # Calculate the mean of all APs
    mAP = np.mean(APs)
    print(f"Mean Average Precision (mAP) @IoU-0.5: {mAP}")

    return mAP


def save_image(image, dest, rois, contours, class_ids, scores):
    import matplotlib.pyplot as plt
    from matplotlib import patches
    from matplotlib.patches import Polygon

    from mrcnn.visualize import random_colors

    # Number of instances
    num_rois = rois.shape[0]

    if not num_rois:
        print("\n*** No instances to display ***\n")

    else:
        assert rois.shape[0] == len(contours) == class_ids.shape[0]

    # Create new axis
    _, ax = plt.subplots(1, figsize=(16, 16))

    # Generate random colors
    colors = random_colors(num_rois)

    # Show area outside image boundaries.
    ax.axis("off")

    for i in range(num_rois):
        color = colors[i]

        # Bounding box
        if not np.any(rois[i]):
            # Skip this instance.
            # Has no bbox. Likely lost in image cropping.
            continue

        y1, x1, y2, x2 = rois[i]

        # Display boundary box (bbox)
        p = patches.Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=2,
            alpha=0.7,
            linestyle="dashed",
            edgecolor=color,
            facecolor="none",
        )
        ax.add_patch(p)

        # Add captions
        class_id = class_ids[i]
        class_names = ["BG", "solar-panel"]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        caption = f"{label} {score:.3f}" if score else label
        ax.text(x1, y1 + 8, caption, color="w", size=11, backgroundcolor="none")

        # Mask Polygon
        for verts in contours:
            # Skip if no verts are available
            if len(verts) == 0:
                continue

            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor=color, edgecolor=color, alpha=0.1)
            ax.add_patch(p)

    ax.imshow(image.astype(np.uint8))

    # Save the plot.
    plt.savefig(dest)


def detect(
    model,
    image_path,
    split=False,
    output_image_path=None,
    output_json_path=None,
):
    """
    Detect solar panel from the given image.

    """
    from skimage.measure import find_contours, approximate_polygon

    # Run model detection and display result
    print(f"Running on {image_path}")

    # Read image
    image = skimage.io.imread(image_path)
    h, w, *_ = image.shape

    # Set block size
    bh, bw = 1024, 1024

    # Crop and adjust color channel of image
    image = image[: bh * (h // bh), : bw * (w // bw), :3]
    h, w, *_ = image.shape

    if split:
        # Split image into blocks
        block_shape = (bh, bw, 3)
        blocks = skimage.util.view_as_blocks(image, block_shape=block_shape)
        cols, rows, *_ = blocks.shape

        # Initialize empty result lists
        all_rois = []
        all_contours = []
        all_class_ids = []
        all_scores = []

        # Iterate every blocks
        for i in range(cols):
            for j in range(rows):
                print(f"working on block({i}, {j})...")

                y = i * bh
                x = j * bw

                # Get the block image
                block = blocks[i, j]

                # Run detection using model
                result = model.detect(block, verbose=1)[0]

                # Skip result digesting if instance does not exist
                if (N := result["rois"].shape[0]) == 0:
                    continue

                # Add offset to result rois
                result["rois"] += np.array([y, x, y, x])

                # Mask Polygon
                for idx in range(N):
                    mask = result["masks"][..., idx]

                    # Pad to ensure proper polygons for masks
                    # that touch image edges.
                    padded_mask = np.zeros(
                        (h + 2, w + 2),
                        dtype=np.uint8,
                    )
                    padded_mask[1 + y : 1 + y + bh, 1 + x : 1 + x + bw] = mask

                    # Find contours from padded mask and simplify it.
                    if contour := find_contours(padded_mask, 0.5):
                        verts = approximate_polygon(contour[0], tolerance=2.0)

                    # TODO: Why this happens? Find out why.
                    else:
                        verts = []

                    # Append vertices to all_verts
                    all_contours.append(verts)

                # Append result to result lists
                all_rois.append(result["rois"])
                all_class_ids.append(result["class_ids"])
                all_scores.append(result["scores"])

        # Return if there's no instances
        if len(all_rois) == 0:
            print("\n*** No instances to display ***\n")
            return

        # Concatenate each result lists
        rois = np.concatenate(all_rois, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)
        scores = np.concatenate(all_scores, axis=0)

        # Use `all_contours` directly
        contours = all_contours

    else:
        # Detect objects
        result = model.detect([image], verbose=1)[0]

        # Initialize variables
        N = result["rois"].shape[0]
        contours = []

        # Mask Polygon
        for idx in range(N):
            mask = result["masks"][..., idx]

            # Pad to ensure proper polygons for masks
            # that touch image edges.
            padded_mask = np.zeros(
                (h + 2, w + 2),
                dtype=np.uint8,
            )
            padded_mask[1:-1, 1:-1] = mask

            # Find contours from padded mask and simplify it.
            contour = find_contours(padded_mask, 0.5)[0]
            verts = approximate_polygon(contour, tolerance=2.0)

            # Append vertices to vertices list
            contours.append(verts)

        # Use result from model directly
        rois = result["rois"]
        class_ids = result["class_ids"]
        scores = result["scores"]

    # Return output image for input image
    if output_image_path:
        masked_image = image.astype(np.uint32).copy()
        save_image(
            masked_image,
            output_image_path,
            rois,
            contours,
            class_ids,
            scores,
        )

    # Return output json for input image
    if output_json_path:
        result_json = {
            "image_id": os.path.basename(image_path),
            "image_size": {
                "width": w,
                "height": h,
            },
            "panel": [],
        }

        for verts in contours:
            # Skip if no verts are available
            if len(verts) == 0:
                continue

            all_points_y, all_points_x = zip(*verts)

            # Add polygon to result json
            result_json["panel"].append(
                {
                    "shape_attributes": {
                        "all_points_x": all_points_x,
                        "all_points_y": all_points_y,
                        "name": "polygon",
                    }
                }
            )

        with open(output_json_path, "w") as fp:
            json.dump(result_json, fp)


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
        "-d",
        required=False,
        metavar="/path/to/solar_panel/datasets/",
        help="Directory of the solar panel datasets",
    )
    parser.add_argument(
        "--weights",
        "-w",
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
        "-i",
        required=False,
        metavar="path or URL to image",
        help="Image to detect solar panel",
    )
    parser.add_argument(
        "--output-image",
        required=False,
        metavar="/path/to/output/image",
        help="Output image file path",
    )
    parser.add_argument(
        "--output-json",
        required=False,
        metavar="/path/to/output/json",
        help="Output json file path",
    )
    parser.add_argument(
        "--split",
        "-s",
        required=False,
        action="store_true",
        help="Enable image splitting",
    )
    args = parser.parse_args()

    # Validate arguments
    match args.command:
        case "train" | "test":
            msg_error = "Dataset is required for training/testing"
            assert args.dataset, msg_error
        case "detect":
            msg_error = "Provide image to apply detection"
            assert args.image, msg_error

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations for training
    if args.command == "train":
        config = SolarPanelConfig()

    # Configurations for testing/detecting
    else:

        class InferenceConfig(SolarPanelConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()

    config.display()

    # Create model for training
    if args.command == "train":
        model = modellib.MaskRCNN(
            mode="training",
            config=config,
            model_dir=args.logs,
        )

    # Create model for testing/detecting
    else:
        model = modellib.MaskRCNN(
            mode="inference",
            config=config,
            model_dir=args.logs,
        )

    # Select weights file to load
    match args.weights.lower():
        case "coco":
            weights_path = COCO_WEIGHTS_PATH

            # Download weights file
            if not os.path.exists(weights_path):
                utils.download_trained_weights(weights_path)
        case "last":
            # Find last trained weights
            weights_path = model.find_last()
        case "imagenet":
            # Start from ImageNet trained weights
            weights_path = model.get_imagenet_weights()
        case _:
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

    # Execute given command
    match args.command:
        case "train":
            train(model)
        case "test":
            mAP = test(model)
            with open("result.log", "a") as f:
                f.write(f"{mAP}\n")

        case "detect":
            detect(
                model,
                image_path=args.image,
                split=args.split,
                output_image_path=args.output_image,
                output_json_path=args.output_json,
            )
        case _:
            print(f"'{args.command}' is not recognized.")
