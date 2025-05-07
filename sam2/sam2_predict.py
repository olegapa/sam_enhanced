import argparse
import os

import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from typing import Any, Dict, List
import cv2

from progress_counter import ProgressCounter
from container_status import ContainerStatus as CS

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


parser = argparse.ArgumentParser(description="Process some images.")

parser.add_argument('--checkpoint', type=str, help='Path to the input image')
parser.add_argument("--input", type=str, help="Flag for training mode")
parser.add_argument("--output", type=str, help="url host with web")
parser.add_argument("--host_web", type=str, help="Flag for demo mode")
parser.add_argument("--total", type=int, help='Path to the input image')
parser.add_argument("--processed", type=int, help='Path to the input image')

args = parser.parse_args()

# Получаем значения аргументов

CHECKPOINT = args.checkpoint
INPUT = args.input
OUTPUT = args.output
HOST_WEB = args.host_web
TOTAL = args.total
PROCESSED = args.processed

predictor = SAM2ImagePredictor(build_sam2(model_cfg, CHECKPOINT))

def write_masks_to_folder(masks, path: str) -> None:
    # header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    # metadata = [header]
    for i, mask_data in enumerate(masks):
        mask = mask_data#["segmentation"]
        filename = f"{i}.png"
        # print("before imwrite")
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        # print(f"after imright, mask = {mask}")
    #     mask_metadata = [
    #         str(i),
    #         str(mask_data["area"]),
    #         *[str(x) for x in mask_data["bbox"]],
    #         *[str(x) for x in mask_data["point_coords"][0]],
    #         str(mask_data["predicted_iou"]),
    #         str(mask_data["stability_score"]),
    #         *[str(x) for x in mask_data["crop_box"]],
    #     ]
    #     row = ",".join(mask_metadata)
    #     metadata.append(row)
    # metadata_path = os.path.join(path, "metadata.csv")
    # with open(metadata_path, "w") as f:
    #     f.write("\n".join(metadata))

    return


with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    cs = CS(args.host_web)
    counter = ProgressCounter(total=int(args.total), processed=int(args.processed), cs=cs)
    targets = [
        f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
    ]
    targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)
    i = 0
    for t in targets:
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)
        masks, _, _ = predictor.predict()

        base = os.path.basename(t)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(args.output, base)

        os.makedirs(save_base, exist_ok=False)
        write_masks_to_folder(masks, save_base)
        i += 1
        # if i % 1000 == 0:
            # counter.report_status(stage=2, report_amount=1000)
    # counter.report_status(stage=2, report_amount=len(targets) % 1000)
    print("Done!")
