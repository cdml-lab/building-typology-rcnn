import numpy as np
import pytest


def test_class_mapping():
    from buildingtypology.classes import (ANALYSIS_CLASSES, NETWORK_CLASSES, analysis_name,
                                          raw_to_network)
    assert len(NETWORK_CLASSES) - 1 == 11          # foreground outputs
    assert len(ANALYSIS_CLASSES) == 10
    assert analysis_name(4) == "H Shape"            # Plus Shape merged
    assert analysis_name(3) == "H Shape"
    assert analysis_name(0) == "None"
    assert raw_to_network(91) == 10 and raw_to_network(3) == 4


def test_iou_and_rotation():
    from buildingtypology.dataset import rotate_boxes
    from buildingtypology.evaluate import iou_matrix
    m = iou_matrix([[0, 0, 10, 10]], [[0, 0, 10, 10], [5, 5, 15, 15]])
    assert m[0, 0] == pytest.approx(1.0) and m[0, 1] == pytest.approx(25 / 175)
    assert rotate_boxes([(1, (200, 200, 300, 300))], 0)[0][1] == pytest.approx((200, 200, 300, 300))


def test_assignment_rule():
    import geopandas as gpd
    from shapely.geometry import box
    from buildingtypology.assign import assign_to_footprints
    fp = gpd.GeoDataFrame(geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(40, 0, 50, 10)],
                          crs="EPSG:32636")
    det = gpd.GeoDataFrame({"network_class_id": [9, 4, 1], "score": [.9, .8, .7]},
                           geometry=[box(0, 0, 10, 10),          # full cover -> A
                                     box(19, 0, 29, 10),         # 90 % cover, Plus -> H
                                     box(40, 0, 41, 10)],        # 10 % cover -> unclassified
                           crs="EPSG:32636")
    out = assign_to_footprints(fp, det)
    assert list(out["class_name"]) == ["Private House_A", "H Shape", "None"]
    assert list(out["network_class"]) == ["Private House_A", "Plus Shape", "None"]


def test_voc_filters(tmp_path):
    from buildingtypology.dataset import read_voc
    xml = tmp_path / "t.xml"
    xml.write_text("<annotation>" + "".join(
        f"<object><name>9</name><bndbox><xmin>0</xmin><ymin>0</ymin><xmax>{s}</xmax>"
        f"<ymax>{s}</ymax></bndbox></object>" for s in (20, 40)) + "</annotation>")
    boxes = read_voc(xml)                       # 20x20 = 400 px2 < 1000 -> dropped
    assert len(boxes) == 1 and boxes[0][0] == 9


def test_manifest_layout(tmp_path):
    from buildingtypology.manifest import build_manifest
    d = tmp_path / "A" / "GOV" / "Unseen"
    (d / "labels").mkdir(parents=True); (d / "images").mkdir()
    (d / "labels" / "t1.xml").write_text(
        "<annotation><size><width>512</width><height>512</height></size><object><name>10</name>"
        "<bndbox><xmin>0</xmin><ymin>0</ymin><xmax>100</xmax><ymax>100</ymax></bndbox></object>"
        "</annotation>")
    (d / "images" / "t1.jgw").write_text("0.4\n0\n0\n-0.4\n1000\n2000\n")
    m = build_manifest(tmp_path, crs="EPSG:2039")
    r = m.iloc[0]
    assert (r.area, r.source, r.split) == ("A", "GOV", "validation")
    assert r.world_file == "A/GOV/Unseen/images/t1.jgw" and not r.image_present
