"""Inference on a georeferenced image (GeoTIFF) of any size.

The image is read in 512 x 512 px windows with a step of 256 px (50 %
overlap). In each window boxes with score >= conf are kept and a
class-agnostic NMS at `tile_nms_iou` is applied; boxes are shifted to image
coordinates; a global class-agnostic NMS at `global_nms_iou` removes the
duplicates created by the overlap. Pixel boxes are converted to map
coordinates with the raster's affine transform and written as polygons.

Edge windows (narrower than 512 px at the right/bottom border):
  edge_mode="pad"     zero-pad to 512 px (geometrically correct; default)
  edge_mode="resize"  stretch to 512 px like the study run (kept so the
                      study's predictions can be reproduced; boxes in such
                      windows are slightly displaced)

The study's imagery had 0.33-0.4 m pixels; use imagery of similar ground
resolution (resample beforehand if necessary).
"""
from __future__ import annotations

import numpy as np

from .classes import analysis_name, network_name

TILE, STEP = 512, 256


def predict_raster(model, device, tif_path, conf=0.3, tile_nms_iou=0.5, global_nms_iou=0.2,
                   edge_mode="pad", bands=(1, 2, 3), log=print):
    import rasterio
    import torch
    import torchvision
    from rasterio.windows import Window
    boxes, labels, scores = [], [], []
    with rasterio.open(tif_path) as src:
        W, H = src.width, src.height
        transform, crs = src.transform, src.crs
        n = 0
        for oy in range(0, max(H - TILE + STEP, 1), STEP):
            for ox in range(0, max(W - TILE + STEP, 1), STEP):
                w, h = min(TILE, W - ox), min(TILE, H - oy)
                if w < 32 or h < 32:
                    continue
                tile = src.read(list(bands), window=Window(ox, oy, w, h))
                if tile.mean() < 2:                      # empty / nodata window
                    continue
                t = torch.from_numpy(tile.astype(np.float32) / 255.0)
                sx = sy = 1.0
                if (w, h) != (TILE, TILE):
                    if edge_mode == "resize":
                        t = torch.nn.functional.interpolate(t[None], size=(TILE, TILE),
                                                            mode="bilinear",
                                                            align_corners=False)[0]
                        sx, sy = w / TILE, h / TILE
                    else:
                        t = torch.nn.functional.pad(t, (0, TILE - w, 0, TILE - h))
                with torch.no_grad():
                    p = model([t.to(device)])[0]
                k = p["scores"] >= conf
                b, l, s = p["boxes"][k], p["labels"][k], p["scores"][k]
                if len(b):
                    keep = torchvision.ops.nms(b, s, tile_nms_iou)
                    b, l, s = b[keep].cpu().numpy(), l[keep].cpu().numpy(), s[keep].cpu().numpy()
                    b[:, [0, 2]] = b[:, [0, 2]] * sx + ox
                    b[:, [1, 3]] = b[:, [1, 3]] * sy + oy
                    b[:, [0, 2]] = b[:, [0, 2]].clip(0, W)
                    b[:, [1, 3]] = b[:, [1, 3]].clip(0, H)
                    boxes.append(b); labels.append(l); scores.append(s)
                n += 1
                if n % 200 == 0:
                    log(f"[infer] {n} windows")
    if not boxes:
        return _to_gdf(np.zeros((0, 4)), np.zeros(0, int), np.zeros(0), transform, crs)
    B = torch.tensor(np.vstack(boxes)); S = torch.tensor(np.concatenate(scores))
    L = np.concatenate(labels)
    keep = torchvision.ops.nms(B, S, global_nms_iou).numpy()
    log(f"[infer] {len(keep)} detections after global NMS")
    return _to_gdf(B.numpy()[keep], L[keep], S.numpy()[keep], transform, crs)


def _to_gdf(b, l, s, transform, crs):
    import geopandas as gpd
    from shapely.geometry import box
    geoms = []
    for x0, y0, x1, y1 in b:
        X0, Y0 = transform * (float(x0), float(y1))
        X1, Y1 = transform * (float(x1), float(y0))
        geoms.append(box(min(X0, X1), min(Y0, Y1), max(X0, X1), max(Y0, Y1)))
    return gpd.GeoDataFrame({
        "network_class_id": np.asarray(l, int),
        "network_class": [network_name(c) for c in l],
        "class_name": [analysis_name(c) for c in l],
        "score": np.asarray(s, float)}, geometry=geoms, crs=crs)
