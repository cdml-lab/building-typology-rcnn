# Method

## Image preparation
Orthophotos are cut into 512 × 512 px RGB tiles (JPEG + ESRI world file,
`.jgw`) at 0.4 m per pixel. Two imagery sources were used for training:
the government orthophoto (GOV) and Esri World Imagery (ESRI), the same
locations in both. Validation uses GOV only.

## Annotation
Pascal-VOC bounding boxes, one per building, with a numeric form code
(`docs/data_format.md`). Codes map to the 11 network classes
(`docs/classes.md`).

## Training and validation areas
Each area (A, B, C) has a training split (`Data`) and a validation split
(`Unseen`). The validation tiles come from areas designated separately
from the training areas before tiling. No validation building is
annotated in any training tile. A few tile footprints touch the training
tiles at the edges of the designated areas, with no shared building. See
`docs/evaluation.md`.

## Inference on a large image (`infer.py`)
* 512 × 512 px windows with a 256 px step (50 % overlap).
* Per window: score ≥ 0.3, class-agnostic NMS at IoU 0.5.
* Boxes shifted to image coordinates; global class-agnostic NMS at IoU 0.2
  removes duplicates from overlapping windows.
* Edge windows, where the image border cuts a window:
  - **Repository default:** zero-padded to 512 px (`--edge-mode pad`).
  - **Study run:** stretched to 512 px (`--edge-mode resize`). This
    displaces boxes slightly in the border strips of the image.

  To reproduce the study's building-form labels, use `--edge-mode resize`
  (`configs/study.yaml`). For new work, keep `pad`.
* Pixel boxes → map polygons with the GeoTIFF's affine transform.

## Assignment to footprints (`assign.py`)
A footprint takes the class of the detection box with the largest
intersection area among boxes covering at least 20 % of the footprint;
otherwise it is unclassified (`None`). No containment filter is applied.

## Evaluation (`evaluate.py`)
Detection recall, classification F1 among matched references and overall
weighted F1 (missed references count as class 0), plus a one-to-one
variant (recall, precision, class accuracy). See `docs/checkpoint.md`.

## Training (`train.py`)
The study setup:
- Data: the training tiles (`Data`) of both imagery sources (GOV and ESRI)
  and all areas. Each tile is used as is plus one copy rotated by a
  random angle.
- Augmentation: colour jitter (p 0.5) and Gaussian blur (p 0.1).
- Optimiser: SGD with lr 0.01, momentum 0.9 and weight decay 5e-4, under a
  one-cycle schedule (warm-up 20 %, cosine).
- Batch 16 with mixed precision; 30 epochs, early stopping with patience 8.
- The epoch is selected on overall weighted F1 on the GOV validation tiles.

Losses: the standard Faster R-CNN losses, with no class weighting.
