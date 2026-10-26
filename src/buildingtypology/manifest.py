"""Dataset manifest: one row per annotated tile, linking the annotation to
its source image, imagery source, split and map footprint."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset import MIN_BOX_AREA_PX, MIN_BOXES_PER_TILE, read_voc, world_file_bounds


def build_manifest(root, area_alias: dict | None = None, crs: str = "") -> pd.DataFrame:
    """area_alias maps folder names to published area labels (e.g. north -> A)."""
    root = Path(root)
    rows = []
    for xml in sorted(root.glob("*/*/*/labels/*.xml")):
        # <root>/<area>/<source>/<split>/labels/<tile>.xml
        area, source, split = xml.parents[3].name, xml.parents[2].name, xml.parents[1].name
        img = xml.parents[1] / "images" / (xml.stem + ".jpg")
        jgw = img.with_suffix(".jgw")
        bounds = world_file_bounds(jgw) if jgw.exists() else (None,) * 4
        boxes_all = read_voc(xml, min_area=0)
        boxes = read_voc(xml, min_area=MIN_BOX_AREA_PX)
        rows.append({
            "area": (area_alias or {}).get(area, area), "source": source,
            "split": "train" if split == "Data" else "validation",
            "annotation": str(xml.relative_to(root)).replace("\\", "/"),
            "image": str(img.relative_to(root)).replace("\\", "/"),
            "image_present": img.exists(),
            "world_file": str(jgw.relative_to(root)).replace("\\", "/") if jgw.exists() else "",
            "xmin": bounds[0], "ymin": bounds[1], "xmax": bounds[2], "ymax": bounds[3],
            "crs": crs, "boxes": len(boxes_all), "boxes_after_filters": len(boxes),
            "used_in_evaluation": split != "Data" and len(boxes) >= MIN_BOXES_PER_TILE,
        })
    return pd.DataFrame(rows)
