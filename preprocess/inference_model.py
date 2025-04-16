import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import (SegformerFeatureExtractor,
                          SegformerForSemanticSegmentation)


@torch.no_grad()
def main():
    depth_model = torch.hub.load('isl-org/ZoeDepth', "ZoeD_NK",
                                 pretrained=True).to('cuda').eval()
    image_processor = SegformerFeatureExtractor.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024")
    seg_model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b5-finetuned-cityscapes-1024-1024").to('cuda').eval()
    image_folder = sys.argv[1]
    depth_folder = image_folder.replace("image", "depth")
    seg_folder = image_folder.replace("image", "semantic")

    os.makedirs(depth_folder, exist_ok=True)
    os.makedirs(seg_folder, exist_ok=True)
    for i, image_path in enumerate(sorted(os.listdir(image_folder))):
        image_path = os.path.join(image_folder, image_path)
        id = image_path.split("/")[-1]
        id = id[:-4]
        image = Image.open(image_path).convert("RGB")
        if not os.path.exists(f"{depth_folder}/{id}.png"):
            depth = depth_model.infer_pil(image, output_type="pil")
            depth.save(f"{depth_folder}/{id}.png")
        if not os.path.exists(f"{seg_folder}/{id}.png"):
            inputs = image_processor(images=image, return_tensors="pt")
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = seg_model(**inputs)
            logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
            logits = torch.nn.functional.interpolate(
                logits,
                size=image.size[::-1],  # (height, width)
                mode="bilinear",
                align_corners=False)
            segmentation = logits.argmax(dim=1)[0].cpu().numpy().astype(
                np.uint8)
            segmentation = Image.fromarray(segmentation)
            segmentation.save(f"{seg_folder}/{id}.png")


if __name__ == '__main__':
    main()
