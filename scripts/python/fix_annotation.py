# run this once on the cluster from your project root
# python scripts/python/fix_annotation.py --species "Prionurus chrysurus"

import json, cv2, numpy as np
from pathlib import Path

SPECIES   = "Prionurus chrysurus"
MASK_PNG  = Path("outputs/val_predictions/Prionurus chrysurus_mask.png")
ANN_JSON  = Path("data/annotations/annotations.json")

# Load predicted mask
mask = cv2.imread(str(MASK_PNG), cv2.IMREAD_GRAYSCALE)
assert mask is not None, f"Mask not found: {MASK_PNG}"

# Extract polygon from predicted mask
cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)  # largest = fish body
epsilon  = 0.001 * cv2.arcLength(cnts[0], True)
approx   = cv2.approxPolyDP(cnts[0], epsilon, True)
polygon  = approx.flatten().tolist()
area     = float(cv2.contourArea(cnts[0]))
x,y,w,h  = cv2.boundingRect(cnts[0])

# Patch annotations.json
with open(ANN_JSON) as f:
    coco = json.load(f)

img_id = next(
    img["id"] for img in coco["images"]
    if SPECIES in img["file_name"]
)

# Find existing annotation for this image and replace it
replaced = False
for ann in coco["annotations"]:
    if ann["image_id"] == img_id:
        ann["segmentation"] = [polygon]
        ann["area"]         = area
        ann["bbox"]         = [x, y, w, h]
        replaced = True
        print(f"Replaced annotation for {SPECIES} (image_id={img_id})")
        break

if not replaced:
    # No annotation existed -- create one
    new_id = max(a["id"] for a in coco["annotations"]) + 1
    coco["annotations"].append({
        "id":           new_id,
        "image_id":     img_id,
        "category_id":  1,
        "segmentation": [polygon],
        "area":         area,
        "bbox":         [x, y, w, h],
        "iscrowd":      0,
    })
    print(f"Added new annotation for {SPECIES} (image_id={img_id})")

with open(ANN_JSON, "w") as f:
    json.dump(coco, f, indent=2)
print("annotations.json updated.")
