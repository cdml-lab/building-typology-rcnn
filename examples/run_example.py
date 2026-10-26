"""Self-contained example: builds a synthetic 1024 x 1024 px orthophoto
(GeoTIFF, 0.4 m pixels, UTM) with drawn buildings and the matching footprint
layer, then runs inference -> assignment and checks the output schema.

    python examples/run_example.py [--checkpoint fasterrcnnv100-12.pth] [--out example_out]

With --checkpoint the trained network is used (synthetic shapes are not
real roofs, so the classes are not meaningful - the run checks the code
path, georeferencing and schema). Without it an untrained network is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_scene(out: Path, seed=0):
    import geopandas as gpd
    import rasterio
    from rasterio.transform import from_origin
    from shapely.geometry import box
    rng = np.random.default_rng(seed)
    px, size = 0.4, 1024
    x0, y0 = 500000.0, 3500000.0                       # UTM 36N, arbitrary
    img = np.full((3, size, size), 150, np.uint8)
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    geoms = []
    for _ in range(40):
        w, h = rng.integers(25, 70, 2)
        c, r = rng.integers(10, size - 80, 2)
        img[:, r:r + h, c:c + w] = rng.integers(60, 230, 3)[:, None, None]
        geoms.append(box(x0 + c * px, y0 - (r + h) * px, x0 + (c + w) * px, y0 - r * px))
    tif = out / "synthetic_ortho.tif"
    with rasterio.open(tif, "w", driver="GTiff", width=size, height=size, count=3,
                       dtype="uint8", crs="EPSG:32636",
                       transform=from_origin(x0, y0, px, px)) as dst:
        dst.write(img)
    fp = gpd.GeoDataFrame({"building_id": range(len(geoms))}, geometry=geoms, crs="EPSG:32636")
    fp.to_file(out / "synthetic_footprints.gpkg")
    return tif, fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint")
    ap.add_argument("--out", default="example_out")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(exist_ok=True)
    from buildingtypology.assign import assign_to_footprints
    from buildingtypology.infer import predict_raster
    from buildingtypology.model import build_model, load_checkpoint
    tif, fp = make_scene(out)
    if a.checkpoint:
        model, dev = load_checkpoint(a.checkpoint)
    else:
        import torch
        dev = torch.device("cpu")
        model = build_model().eval()
        print("no checkpoint: untrained network (code-path check only)")
    det = predict_raster(model, dev, tif, conf=0.3 if a.checkpoint else 0.0)
    det.to_file(out / "detections.gpkg")
    lab = assign_to_footprints(fp, det)
    lab.to_file(out / "footprints_typology.gpkg")
    need = {"network_class_id", "network_class", "class_name", "score", "coverage"}
    assert need <= set(lab.columns), "output schema incomplete"
    assert set(lab["class_name"]) <= {"None", "Rectangle Building", "Long Rectangle", "H Shape",
                                      "L Shape", "Geometric Building", "Geometric Shape",
                                      "Carpet Houses", "Private House_A", "Private House_B",
                                      "U Shape"}, "unexpected class name"
    print(f"{len(det)} detections; {(lab['class_name'] != 'None').sum()} of {len(lab)} "
          f"footprints classified -> {out / 'footprints_typology.gpkg'}")
    print("example OK")


if __name__ == "__main__":
    main()
