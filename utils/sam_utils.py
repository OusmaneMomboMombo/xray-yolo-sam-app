import torch
import numpy as np
import cv2
import base64
from segment_anything import sam_model_registry, SamPredictor


###################################
# 1. CHARGER SAM
###################################

def load_sam_model(model_path="models/sam_vit_b.pth", model_type="vit_b", device="cpu"):
    sam = sam_model_registry[model_type](checkpoint=model_path)
    sam.to(device=device)
    return SamPredictor(sam)


###################################
# 2. SAM via bounding box YOLO
###################################

def sam_from_bbox(predictor, image_path, bbox):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    masks, _, _ = predictor.predict(
        box=np.array(bbox),
        multimask_output=False
    )
    return masks[0]


###################################
# 3. SAM via clic utilisateur
###################################

def sam_from_point(predictor, image_path, point):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image_rgb)

    masks, _, _ = predictor.predict(
        point_coords=np.array([point]),
        point_labels=np.array([1]),
        multimask_output=False
    )
    return masks[0]


###################################
# 4. Convertir masque → Base64
###################################

def mask_to_base64(mask):
    mask_img = (mask * 255).astype(np.uint8)
    _, buffer = cv2.imencode(".png", mask_img)
    return base64.b64encode(buffer).decode("utf-8")
