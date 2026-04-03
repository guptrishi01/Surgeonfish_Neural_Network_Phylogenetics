# -*- coding: utf-8 -*-
"""
train_mask_rcnn.py

Fine-tunes a pretrained Mask R-CNN (ResNet-50 + FPN) on the surgeonfish
dataset using torchvision. Designed for the iterative ML pipeline:

  Iteration 1: train on 43 images, evaluate on 10 val images
  Review:       inspect predicted masks in outputs/val_predictions/
  Adjust:       move any misannotated images between splits if needed,
                update annotations.json, re-run prepare_splits.py
  Repeat:       until val masks are consistently correct
  Final eval:   run once on test set -- do not look at test until then

Workflow flags:
  --mode train    fine-tune the model, save best checkpoint
  --mode val      run inference on val set, save mask PNGs for inspection
  --mode test     final evaluation on test set (run once at the very end)
  --mode predict  run inference on a single image

Output structure:
  outputs/
    checkpoints/
      best_model.pth         <- best val mask AP checkpoint
      checkpoint_epNNN.pth   <- checkpoint every N epochs
    val_predictions/
      <species>_mask.png     <- predicted binary mask
      <species>_overlay.png  <- image with green=predicted, red=ground truth
    test_predictions/        <- same structure, written only in --mode test
    training_log.csv         <- epoch, train_loss, val_mask_AP per row

Usage:
  python scripts/python/train_mask_rcnn.py --mode train
  python scripts/python/train_mask_rcnn.py --mode val
  python scripts/python/train_mask_rcnn.py --mode test
  python scripts/python/train_mask_rcnn.py --mode predict \
      --image "data/standardized_images/Naso/Naso annulatus.png"
  python scripts/python/train_mask_rcnn.py --mode train \
      --resume outputs/checkpoints/checkpoint_ep010.pth
  python scripts/python/train_mask_rcnn.py --mode train --unfreeze-backbone
"""

import copy
import csv
import json
import logging
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

import torch
import torch.utils.data
import torchvision.transforms.functional as TF
from torchvision.models.detection import (
    maskrcnn_resnet50_fpn,
    MaskRCNN_ResNet50_FPN_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR   = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR     = PROJECT_ROOT / "data"
ANN_DIR      = DATA_DIR / "annotations"
STD_DIR      = DATA_DIR / "standardized_images"
OUT_DIR      = PROJECT_ROOT / "outputs"

ANN_JSON       = ANN_DIR / "annotations.json"
TRAIN_IDS_FILE = ANN_DIR / "train_ids.txt"
VAL_IDS_FILE   = ANN_DIR / "val_ids.txt"
TEST_IDS_FILE  = ANN_DIR / "test_ids.txt"

CHECKPOINT_DIR = OUT_DIR / "checkpoints"
VAL_PRED_DIR   = OUT_DIR / "val_predictions"
TEST_PRED_DIR  = OUT_DIR / "test_predictions"
LOG_PATH       = OUT_DIR / "training_log.csv"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

NUM_CLASSES    = 2      # 0=background, 1=fish
NUM_EPOCHS     = 50
BATCH_SIZE     = 2      # safe for 11 GB VRAM at 1024x1024
LEARNING_RATE  = 0.005
MOMENTUM       = 0.9
WEIGHT_DECAY   = 0.0005
LR_STEP_SIZE   = 15     # decay LR every N epochs
LR_GAMMA       = 0.1
CHECKPOINT_FREQ= 5      # save checkpoint every N epochs
MASK_THRESHOLD = 0.5    # binarise soft mask predictions
SCORE_THRESHOLD= 0.5    # minimum detection confidence to keep

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SurgeonFishDataset(torch.utils.data.Dataset):
    """
    COCO-format dataset for surgeonfish segmentation.

    Returns per item:
      image  : FloatTensor [3, H, W]  values in [0, 1]
      target : dict
        image_id : Tensor scalar
        boxes    : FloatTensor [N, 4]  xyxy
        labels   : LongTensor  [N]     all 1 (fish)
        masks    : BoolTensor  [N, H, W]
        area     : FloatTensor [N]
        iscrowd  : LongTensor  [N]

    Images with no annotations (e.g. Naso annulatus) return empty targets --
    skipped during training, used for inference.
    """

    def __init__(
        self,
        coco_json: Path,
        image_ids: List[int],
        image_root: Path,
        augment: bool = False,
    ):
        with open(coco_json) as f:
            self.coco = json.load(f)

        self.image_root = image_root
        self.augment    = augment

        self.id_to_image = {img["id"]: img for img in self.coco["images"]}

        self.id_to_anns: Dict[int, list] = {
            img["id"]: [] for img in self.coco["images"]
        }
        for ann in self.coco["annotations"]:
            self.id_to_anns[ann["image_id"]].append(ann)

        self.image_ids = [i for i in image_ids if i in self.id_to_image]
        logger.info(
            f"Dataset: {len(self.image_ids)} images "
            f"({'train+augment' if augment else 'eval'})"
        )

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        import random

        img_id   = self.image_ids[idx]
        img_info = self.id_to_image[img_id]
        anns     = self.id_to_anns[img_id]

        img_path = self.image_root / img_info["file_name"]
        image    = Image.open(img_path).convert("RGB")

        # Augmentation (training only)
        # Only horizontal flip + brightness/contrast jitter.
        # No rotation -- fish orientation matters for morphometrics.
        hflip = False
        if self.augment:
            if random.random() > 0.5:
                image = TF.hflip(image)
                hflip = True
            image = TF.adjust_brightness(image, 1.0 + random.uniform(-0.2,  0.2))
            image = TF.adjust_contrast(image,   1.0 + random.uniform(-0.15, 0.15))

        image_tensor = TF.to_tensor(image)
        H, W = image_tensor.shape[1], image_tensor.shape[2]

        empty_target = {
            "image_id": torch.tensor(img_id),
            "boxes":    torch.zeros((0, 4), dtype=torch.float32),
            "labels":   torch.zeros(0, dtype=torch.int64),
            "masks":    torch.zeros((0, H, W), dtype=torch.bool),
            "area":     torch.zeros(0, dtype=torch.float32),
            "iscrowd":  torch.zeros(0, dtype=torch.int64),
        }

        if not anns:
            return image_tensor, empty_target

        masks, boxes, areas = [], [], []
        for ann in anns:
            mask = self._polygon_to_mask(ann["segmentation"], H, W)
            if mask.sum() == 0:
                continue
            if hflip:
                mask = np.fliplr(mask).copy()
            masks.append(mask)
            x, y, bw, bh = ann["bbox"]
            if hflip:
                x = W - (x + bw)
            boxes.append([x, y, x + bw, y + bh])
            areas.append(float(ann["area"]))

        if not masks:
            return image_tensor, empty_target

        target = {
            "image_id": torch.tensor(img_id),
            "boxes":    torch.as_tensor(boxes, dtype=torch.float32),
            "labels":   torch.ones(len(masks), dtype=torch.int64),
            "masks":    torch.as_tensor(np.stack(masks), dtype=torch.bool),
            "area":     torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd":  torch.zeros(len(masks), dtype=torch.int64),
        }
        return image_tensor, target

    def _polygon_to_mask(
        self, segmentation: list, H: int, W: int
    ) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.uint8)
        for poly in segmentation:
            pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
            cv2.fillPoly(mask, [pts], 1)
        return mask.astype(bool)


def collate_fn(batch):
    return tuple(zip(*batch))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int = NUM_CLASSES) -> torch.nn.Module:
    model = maskrcnn_resnet50_fpn(
        weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features_box, num_classes
    )
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )
    return model


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def compute_mask_ap(
    predictions: list,
    targets: list,
    iou_threshold: float = 0.5,
) -> float:
    aps = []
    for pred, tgt in zip(predictions, targets):
        gt_masks = tgt["masks"].numpy() if tgt["masks"].numel() > 0 else []
        if len(gt_masks) == 0:
            continue   # no ground truth -- skip (Naso annulatus)

        pred_masks  = pred.get("masks",  [])
        pred_scores = pred.get("scores", [])

        if len(pred_masks) == 0:
            aps.append(0.0)
            continue

        best_idx = np.argmax(pred_scores)
        pred_bin = (pred_masks[best_idx] > MASK_THRESHOLD)
        gt_bin   = gt_masks[0].astype(bool)
        aps.append(1.0 if mask_iou(pred_bin, gt_bin) >= iou_threshold else 0.0)

    return float(np.mean(aps)) if aps else 0.0


# ---------------------------------------------------------------------------
# Train / eval loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, optimizer, loader, device, epoch
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for images, targets in loader:
        if all(t["boxes"].numel() == 0 for t in targets):
            continue
        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses    = sum(loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        total_loss += losses.item()
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.inference_mode()
def evaluate(
    model, loader, device
) -> Tuple[float, list, list]:
    model.eval()
    all_preds, all_tgts = [], []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        preds  = model(images)
        for pred in preds:
            all_preds.append({
                "masks":  pred["masks"].squeeze(1).cpu().numpy(),
                "scores": pred["scores"].cpu().numpy(),
                "boxes":  pred["boxes"].cpu().numpy(),
                "labels": pred["labels"].cpu().numpy(),
            })
        for tgt in targets:
            all_tgts.append({k: v.cpu() for k, v in tgt.items()})

    return compute_mask_ap(all_preds, all_tgts), all_preds, all_tgts


# ---------------------------------------------------------------------------
# Save predictions for visual inspection
# ---------------------------------------------------------------------------

def save_predictions(
    predictions: list,
    targets: list,
    dataset: SurgeonFishDataset,
    out_dir: Path,
) -> None:
    """
    For each image writes two files:
      <species>_mask.png     binary mask  (white = fish)
      <species>_overlay.png  original image with:
                               green fill  = predicted mask
                               red contour = ground truth boundary
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    for pred, tgt in zip(predictions, targets):
        img_id   = int(tgt["image_id"])
        img_info = dataset.id_to_image.get(img_id)
        if img_info is None:
            continue

        species  = Path(img_info["file_name"]).stem
        img_path = dataset.image_root / img_info["file_name"]
        image_np = np.array(Image.open(img_path).convert("RGB"))
        H, W     = image_np.shape[:2]

        # Predicted binary mask
        pred_masks  = pred.get("masks",  np.array([]))
        pred_scores = pred.get("scores", np.array([]))
        valid       = pred_scores >= SCORE_THRESHOLD

        if valid.any():
            best     = pred_scores[valid].argmax()
            bin_mask = (pred_masks[valid][best] > MASK_THRESHOLD).astype(np.uint8) * 255
        else:
            bin_mask = np.zeros((H, W), dtype=np.uint8)

        Image.fromarray(bin_mask, mode="L").save(
            str(out_dir / f"{species}_mask.png")
        )

        # Overlay: green predicted + red ground truth boundary
        overlay     = image_np.copy()
        green_layer = np.zeros_like(image_np)
        green_layer[bin_mask > 127] = [0, 200, 0]
        overlay = cv2.addWeighted(overlay, 0.75, green_layer, 0.25, 0)

        gt_masks = tgt["masks"].numpy() if tgt["masks"].numel() > 0 else []
        if len(gt_masks) > 0:
            gt_bin = (gt_masks[0] > 0).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(
                gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, cnts, -1, (255, 0, 0), 2)

        Image.fromarray(overlay).save(
            str(out_dir / f"{species}_overlay.png")
        )

    logger.info(f"Saved {len(predictions)} prediction pairs -> {out_dir}")


# ---------------------------------------------------------------------------
# Mode: train
# ---------------------------------------------------------------------------

def run_training(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    train_ids = [int(x) for x in TRAIN_IDS_FILE.read_text().split()]
    val_ids   = [int(x) for x in VAL_IDS_FILE.read_text().split()]

    # Build datasets
    train_ds_full = SurgeonFishDataset(ANN_JSON, train_ids, STD_DIR, augment=True)

    # Filter unannotated images out of the training loader
    annotated_train_ids = [
        i for i in train_ids
        if len(train_ds_full.id_to_anns.get(i, [])) > 0
    ]
    train_ds = SurgeonFishDataset(
        ANN_JSON, annotated_train_ids, STD_DIR, augment=True
    )
    val_ds   = SurgeonFishDataset(ANN_JSON, val_ids, STD_DIR, augment=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    model = build_model(NUM_CLASSES).to(device)

    # Freeze backbone by default -- recommended for small datasets.
    # Use --unfreeze-backbone only if val AP plateaus after many epochs.
    if args.unfreeze_backbone:
        params = [p for p in model.parameters() if p.requires_grad]
        logger.info("Training full model (backbone + heads)")
    else:
        for name, param in model.named_parameters():
            if "backbone" in name:
                param.requires_grad = False
        params = [p for p in model.parameters() if p.requires_grad]
        logger.info("Backbone frozen -- training heads only")

    optimizer = torch.optim.SGD(
        params, lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    start_epoch = 1
    best_val_ap = 0.0
    best_weights= copy.deepcopy(model.state_dict())

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_ap = ckpt.get("best_val_ap", 0.0)
        logger.info(f"Resumed from {args.resume} at epoch {ckpt['epoch']}")

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log_rows = []
    if LOG_PATH.exists() and args.resume:
        with open(LOG_PATH) as f:
            log_rows = list(csv.DictReader(f))

    logger.info("")
    logger.info("=" * 70)
    logger.info("TRAINING")
    logger.info("=" * 70)
    logger.info(f"Train images : {len(annotated_train_ids)}")
    logger.info(f"Val images   : {len(val_ids)}")
    logger.info(f"Epochs       : {NUM_EPOCHS}  (start={start_epoch})")
    logger.info(f"Batch size   : {BATCH_SIZE}")
    logger.info(f"LR           : {LEARNING_RATE}")
    logger.info("")

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss = train_one_epoch(model, optimizer, train_loader, device, epoch)
        scheduler.step()
        val_ap, val_preds, val_tgts = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
            f"loss={train_loss:.4f}  "
            f"val_mask_AP={val_ap:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  "
            f"({elapsed:.1f}s)"
        )

        log_rows.append({
            "epoch":       epoch,
            "train_loss":  round(train_loss, 5),
            "val_mask_AP": round(val_ap, 5),
            "lr":          round(scheduler.get_last_lr()[0], 8),
            "elapsed_s":   round(elapsed, 1),
        })
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_rows[0].keys())
            writer.writeheader()
            writer.writerows(log_rows)

        # Save best checkpoint
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_weights= copy.deepcopy(model.state_dict())
            torch.save({
                "epoch":           epoch,
                "model_state":     best_weights,
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_ap":     best_val_ap,
                "val_mask_AP":     val_ap,
            }, CHECKPOINT_DIR / "best_model.pth")
            logger.info(f"  -> New best  val_mask_AP={best_val_ap:.4f}")

        # Periodic checkpoint
        if epoch % CHECKPOINT_FREQ == 0:
            torch.save({
                "epoch":           epoch,
                "model_state":     model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_ap":     best_val_ap,
                "val_mask_AP":     val_ap,
            }, CHECKPOINT_DIR / f"checkpoint_ep{epoch:03d}.pth")

    logger.info("")
    logger.info(f"Training complete. Best val mask AP: {best_val_ap:.4f}")
    logger.info(f"Best model: {CHECKPOINT_DIR / 'best_model.pth'}")
    logger.info("Next: python scripts/python/train_mask_rcnn.py --mode val")


# ---------------------------------------------------------------------------
# Mode: val / test
# ---------------------------------------------------------------------------

def run_inference(args, split: str):
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids_file = VAL_IDS_FILE  if split == "val"  else TEST_IDS_FILE
    out_dir  = VAL_PRED_DIR  if split == "val"  else TEST_PRED_DIR
    ids      = [int(x) for x in ids_file.read_text().split()]

    dataset = SurgeonFishDataset(ANN_JSON, ids, STD_DIR, augment=False)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    model     = build_model(NUM_CLASSES).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint \
                else CHECKPOINT_DIR / "best_model.pth"

    if not ckpt_path.exists():
        logger.error(f"No checkpoint at {ckpt_path}. Run --mode train first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(
        f"Loaded: {ckpt_path}  "
        f"(epoch={ckpt['epoch']}, val_AP={ckpt.get('val_mask_AP', 0):.4f})"
    )

    mask_ap, preds, tgts = evaluate(model, loader, device)
    logger.info(f"{split.upper()} mask AP @ IoU=0.5 : {mask_ap:.4f}")
    save_predictions(preds, tgts, dataset, out_dir)

    if split == "val":
        logger.info("")
        logger.info("Inspect outputs/val_predictions/")
        logger.info("  <species>_overlay.png  green=predicted  red=ground truth")
        logger.info("")
        logger.info("If a mask is wrong:")
        logger.info("  - Bad annotation -> fix annotations.json, re-run prepare_splits.py")
        logger.info("  - Bad prediction but good annotation -> train more epochs")
    else:
        logger.info(f"Final test mask AP: {mask_ap:.4f}")


# ---------------------------------------------------------------------------
# Mode: predict (single image)
# ---------------------------------------------------------------------------

@torch.inference_mode()
def run_predict(args):
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = build_model(NUM_CLASSES).to(device)
    ckpt_path = Path(args.checkpoint) if args.checkpoint \
                else CHECKPOINT_DIR / "best_model.pth"

    if not ckpt_path.exists():
        logger.error(f"No checkpoint at {ckpt_path}. Run --mode train first.")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    img_path = Path(args.image)
    image_np = np.array(Image.open(img_path).convert("RGB"))
    tensor   = TF.to_tensor(Image.fromarray(image_np)).unsqueeze(0).to(device)

    pred    = model(tensor)[0]
    scores  = pred["scores"].cpu().numpy()
    masks   = pred["masks"].squeeze(1).cpu().numpy()
    valid   = scores >= SCORE_THRESHOLD

    if valid.any():
        best     = scores[valid].argmax()
        bin_mask = (masks[valid][best] > MASK_THRESHOLD).astype(np.uint8) * 255
    else:
        logger.warning("No detections above score threshold.")
        bin_mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

    out_dir = OUT_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    species = img_path.stem

    Image.fromarray(bin_mask, mode="L").save(
        str(out_dir / f"{species}_mask.png")
    )

    overlay     = image_np.copy()
    green_layer = np.zeros_like(image_np)
    green_layer[bin_mask > 127] = [0, 200, 0]
    overlay = cv2.addWeighted(overlay, 0.75, green_layer, 0.25, 0)
    Image.fromarray(overlay).save(
        str(out_dir / f"{species}_overlay.png")
    )

    logger.info(f"Mask:    {out_dir / f'{species}_mask.png'}")
    logger.info(f"Overlay: {out_dir / f'{species}_overlay.png'}")
    if valid.any():
        logger.info(f"Score:   {scores[valid].max():.4f}")
        logger.info(f"Area:    {(bin_mask > 127).mean():.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    global NUM_EPOCHS
    parser = argparse.ArgumentParser(
        description="Train and evaluate Mask R-CNN on surgeonfish dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["train", "val", "test", "predict"],
        default="train",
    )
    parser.add_argument("--resume",    type=str, default=None,
                        help="Checkpoint path to resume training from")
    parser.add_argument("--checkpoint",type=str, default=None,
                        help="Checkpoint for val/test/predict (default: best_model.pth)")
    parser.add_argument("--image",     type=str, default=None,
                        help="Image path for --mode predict")
    parser.add_argument("--unfreeze-backbone", action="store_true",
                        help="Train full model including backbone")
    parser.add_argument("--epochs",    type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    if args.mode == "train":
        run_training(args)
    elif args.mode == "val":
        run_inference(args, "val")
    elif args.mode == "test":
        logger.warning("=" * 70)
        logger.warning("TEST SET EVALUATION")
        logger.warning("Run this ONCE only at the very end.")
        logger.warning("Do not use test results to guide further training.")
        logger.warning("=" * 70)
        run_inference(args, "test")
    elif args.mode == "predict":
        if not args.image:
            parser.error("--image is required for --mode predict")
        run_predict(args)


if __name__ == "__main__":
    main()
