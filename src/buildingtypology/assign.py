"""Assign detections to building footprints.

A footprint takes the class of the detection box with the LARGEST absolute
intersection area, among boxes that cover at least `coverage` (default 20 %)
of the footprint's area. Footprints without such a box are unclassified
(class_name "None"). One box may label several footprints; each footprint
gets at most one label. No containment filter is applied (as in the study).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .classes import UNCLASSIFIED, analysis_name, network_name


def assign_to_footprints(footprints, detections, coverage: float = 0.20,
                         metric_crs=None):
    """footprints: GeoDataFrame of building polygons (any columns kept).
    detections: output of infer.predict_raster. Returns the footprints with
    network_class_id, network_class, class_name (10 analysis classes),
    score and coverage columns (see docs/output_schema.md)."""
    import geopandas as gpd
    b = footprints.copy()
    crs = metric_crs or (b.crs if b.crs and not b.crs.is_geographic else b.estimate_utm_crs())
    bm = b.to_crs(crs).reset_index(drop=True)
    dm = detections.to_crs(crs).reset_index(drop=True)
    bm["geometry"] = bm.geometry.buffer(0)
    out = b.reset_index(drop=True)
    out["network_class_id"] = 0
    out["network_class"] = UNCLASSIFIED
    out["class_name"] = UNCLASSIFIED
    out["score"] = np.nan
    out["coverage"] = np.nan
    if len(dm) == 0 or len(bm) == 0:
        return out
    pairs = gpd.sjoin(dm[["geometry", "network_class_id", "score"]],
                      bm[["geometry"]], how="inner", predicate="intersects")
    if pairs.empty:
        return out
    bi = pairs["index_right"].to_numpy()
    inter = gpd.GeoSeries(pairs.geometry.values, crs=crs).intersection(
        bm.geometry.iloc[bi].reset_index(drop=True)).area.to_numpy()
    area = bm.geometry.area.to_numpy()[bi]
    cov = np.divide(inter, area, out=np.zeros_like(inter), where=area > 0)
    cand = pd.DataFrame({"b": bi, "inter": inter, "cov": cov,
                         "cid": pairs["network_class_id"].to_numpy(),
                         "score": pairs["score"].to_numpy()})
    cand = cand[(cand["cov"] >= coverage) & (cand["inter"] > 0)]
    if cand.empty:
        return out
    best = cand.loc[cand.groupby("b")["inter"].idxmax()]
    out.loc[best["b"], "network_class_id"] = best["cid"].to_numpy()
    out.loc[best["b"], "network_class"] = [network_name(c) for c in best["cid"]]
    out.loc[best["b"], "class_name"] = [analysis_name(c) for c in best["cid"]]
    out.loc[best["b"], "score"] = best["score"].to_numpy()
    out.loc[best["b"], "coverage"] = best["cov"].to_numpy()
    return out
