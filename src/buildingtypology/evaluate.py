"""Validation metrics, as defined in the study.

Per 512-px validation tile the detector's boxes are kept at score >= conf,
then class-agnostic NMS at `iou` is applied, and every reference box is
matched to the prediction of highest IoU (>= iou):

  detection recall       matched references / all references
  classification F1      weighted F1 of (reference class, predicted class)
                         over the matched references
  overall weighted F1    weighted F1 over all references, a missed
                         reference counting as predicted class 0
  (one-to-one)           greedy one-to-one matching at IoU >= iou: recall,
                         precision, class accuracy among detected

`classes="analysis"` maps references and predictions through the merge of
Plus Shape into H Shape (10 classes) before scoring; `classes="network"`
scores the 11 network outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .classes import NETWORK_TO_ANALYSIS
from .dataset import MIN_BOXES_PER_TILE, collect_paths, read_voc, world_file_bounds


def predict_tile(model, device, img, conf=0.3, nms_iou=0.5):
    """Boxes, network class ids and scores for one RGB PIL tile."""
    import torch
    import torchvision
    x = torchvision.transforms.ToTensor()(img).to(device)
    with torch.no_grad():
        p = model([x])[0]
    k = p["scores"] >= conf
    b, s, lab = p["boxes"][k], p["scores"][k], p["labels"][k]
    keep = torchvision.ops.nms(b, s, nms_iou)
    return (b[keep].cpu().numpy(), lab[keep].cpu().numpy().astype(int),
            s[keep].cpu().numpy())


def iou_matrix(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + bb[None, :] - inter)


def overlapping_tiles(train_dir, val_dir):
    """Validation tile stems whose footprint intersects any training tile
    (from the .jgw world files)."""
    from shapely.geometry import box
    from shapely.strtree import STRtree
    tr = [box(*world_file_bounds(p)) for p in Path(train_dir, "images").glob("*.jgw")]
    if not tr:
        return set()
    tree = STRtree(tr)
    out = set()
    for p in Path(val_dir, "images").glob("*.jgw"):
        g = box(*world_file_bounds(p))
        if any(g.intersection(tr[i]).area > 0 for i in tree.query(g)):
            out.add(p.stem)
    return out


def evaluate(model, device, root, areas=None, source="GOV", split="Unseen", conf=0.3,
             iou=0.5, classes="analysis", spatially_separate=False):
    from PIL import Image
    from sklearn.metrics import precision_recall_fscore_support as prf
    imgs, labs = collect_paths(root, areas, (source,), split)
    excl = set()
    if spatially_separate:
        for a in {Path(p).parents[3].name for p in imgs}:
            excl |= {(a, s) for s in overlapping_tiles(Path(root, a, source, "Data"),
                                                       Path(root, a, source, split))}
    mg = (lambda c: NETWORK_TO_ANALYSIS.get(int(c), int(c))) if classes == "analysis" \
        else (lambda c: int(c))
    per = {}
    for ip, lp in zip(imgs, labs):
        area = Path(ip).parents[3].name
        if (area, Path(ip).stem) in excl:
            continue
        refs = read_voc(lp)
        if len(refs) < MIN_BOXES_PER_TILE:
            continue
        pb, pl, _ = predict_tile(model, device, Image.open(ip).convert("RGB"), conf, iou)
        rb = np.array([b for _, b in refs], float)
        rc = [mg(c) for c, _ in refs]
        d = per.setdefault(area, {"tiles": 0, "refs": 0, "matched": 0, "preds": 0, "o2o": 0,
                                  "o2o_correct": 0, "at": [], "ap": [], "mt": [], "mp": []})
        d["tiles"] += 1
        d["refs"] += len(refs)
        d["preds"] += len(pb)
        m = iou_matrix(rb, pb)
        if len(pb):
            best, arg = m.max(1), m.argmax(1)
        for i in range(len(refs)):
            d["at"].append(rc[i])
            if len(pb) and best[i] >= iou:
                q = mg(pl[arg[i]])
                d["ap"].append(q); d["mt"].append(rc[i]); d["mp"].append(q); d["matched"] += 1
            else:
                d["ap"].append(0)
        # greedy one-to-one
        pairs = sorted(((m[i, j], i, j) for i in range(m.shape[0]) for j in range(m.shape[1])
                        if m[i, j] >= iou), reverse=True)
        ui, uj = set(), set()
        for _, i, j in pairs:
            if i not in ui and j not in uj:
                ui.add(i); uj.add(j)
                d["o2o"] += 1
                d["o2o_correct"] += int(rc[i] == mg(pl[j]))
    rows = []

    def row(name, ds):
        at = sum((d["at"] for d in ds), []); ap = sum((d["ap"] for d in ds), [])
        mt = sum((d["mt"] for d in ds), []); mp = sum((d["mp"] for d in ds), [])
        tot = lambda k: sum(d[k] for d in ds)
        f = lambda a, b: 100 * prf(a, b, average="weighted", zero_division=0)[2] if a else np.nan
        return {"set": name, "tiles": tot("tiles"), "references": tot("refs"),
                "predictions": tot("preds"),
                "detection_recall": 100 * tot("matched") / max(tot("refs"), 1),
                "classification_F1": f(mt, mp), "overall_weighted_F1": f(at, ap),
                "o2o_recall": 100 * tot("o2o") / max(tot("refs"), 1),
                "o2o_precision": 100 * tot("o2o") / max(tot("preds"), 1),
                "o2o_class_accuracy": 100 * tot("o2o_correct") / max(tot("o2o"), 1)}
    for a in sorted(per):
        rows.append(row(a, [per[a]]))
    rows.append(row("all", list(per.values())))
    out = pd.DataFrame(rows)
    out.attrs.update({"conf": conf, "iou": iou, "classes": classes, "source": source,
                      "split": split, "spatially_separate": spatially_separate,
                      "excluded_overlapping_tiles": len(excl)})
    return out


def write_report(df, path):
    path = Path(path)
    df.round(2).to_csv(path.with_suffix(".csv"), index=False)
    path.with_suffix(".json").write_text(json.dumps(
        {"configuration": df.attrs, "metrics": df.round(4).to_dict(orient="records")}, indent=1))
