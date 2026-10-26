# Released checkpoint

| item | value |
|---|---|
| file | `fasterrcnnv100-12.pth` (PyTorch `state_dict`, 242 MB) |
| version | v100-12, trained February 2026 |
| architecture | torchvision `FasterRCNN`, `resnet_fpn_backbone("resnet101")`, 12 outputs (background + 11) |
| SHA-256 | `87727651374c5ae9e2b652f4c01487d1565b9971bff670044b86733604c1dc78` |
| distribution | release asset of v1.0.0 (https://github.com/cdml-lab/building-typology-rcnn/releases/download/v1.0.0/fasterrcnnv100-12.pth) and the results archive; not in Git history; the training imagery is not included |
| load | `buildingtypology.model.load_checkpoint(path)` (checks the SHA-256) |

## Evaluation configuration (study)

* Validation tiles: split `Unseen`, imagery source GOV, three areas; tiles
  of 512 × 512 px at 0.4 m; boxes < 1000 px² dropped; tiles with < 10
  boxes dropped → 255 tiles, 6,440 annotated building instances.
* Score threshold 0.3; class-agnostic NMS at IoU 0.5; reference–prediction
  matching at IoU ≥ 0.5 (see `evaluate.py` for the metric definitions).
* The validation tiles (from areas selected separately from training
  before tiling) informed the selection of the training epoch and of the
  thresholds, so they are a validation set, not an independent test set.

| set (GOV, 10 analysis classes, IoU 0.5) | tiles | detection recall % | classification F1 % | overall weighted F1 % |
|---|---|---|---|---|
| all | 255 | 82.34 | 75.59 | 67.05 |

The full table (each city, Esri imagery, IoU 0.4), the metric definitions,
the sample sizes and the check of the validation/training separation are
in `docs/evaluation.md`. The results were reproduced with this repository
(`buildingtypology evaluate`, torch 2.7.1, torchvision 0.22.1).
