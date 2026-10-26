"""Building-form categories.

Three label spaces are involved:

1. RAW annotation codes - the numeric <name> values in the Pascal-VOC
   annotations (several historical codes per form).
2. NETWORK classes - the 11 foreground outputs of the trained detector
   (+ background 0). RAW codes map to NETWORK ids through RAW_TO_NETWORK.
3. ANALYSIS classes - the 10 building-form categories used in the block
   morphology analysis: NETWORK class 4 "Plus Shape" (rare) is merged into
   3 "H Shape" after prediction (NETWORK_TO_ANALYSIS).
"""
from __future__ import annotations

# 2 -> 1: raw annotation code -> network class id (0 = background)
RAW_TO_NETWORK: dict[int, int] = {
    1: 1,
    11: 2, 12: 2, 101: 2, 102: 2,
    2: 3,
    3: 4,
    4: 5,
    7: 6, 20: 6,
    5: 7, 6: 7,
    8: 8, 81: 8,
    9: 9,
    91: 10, 92: 10,
    10: 11,
}

# network outputs (index = class id the model predicts)
NETWORK_CLASSES: dict[int, str] = {
    0: "Background",
    1: "Rectangle Building",
    2: "Long Rectangle",
    3: "H Shape",
    4: "Plus Shape",
    5: "L Shape",
    6: "Geometric Building",
    7: "Geometric Shape",
    8: "Carpet Houses",
    9: "Private House_A",
    10: "Private House_B",
    11: "U Shape",
}
NUM_NETWORK_CLASSES = 12            # background + 11 (the checkpoint's head size)

# 3: merge used in the morphology analysis
NETWORK_TO_ANALYSIS: dict[int, int] = {i: i for i in range(12)}
NETWORK_TO_ANALYSIS[4] = 3          # Plus Shape -> H Shape

ANALYSIS_CLASSES: dict[int, str] = {
    1: "Rectangle Building",
    2: "Long Rectangle",
    3: "H Shape",
    5: "L Shape",
    6: "Geometric Building",
    7: "Geometric Shape",
    8: "Carpet Houses",
    9: "Private House_A",
    10: "Private House_B",
    11: "U Shape",
}
UNCLASSIFIED = "None"               # class_name of a footprint without a detection

# column names used by the morphology repository (shares per block)
ANALYSIS_COLUMNS: dict[str, str] = {
    name: "rcnn_" + name.lower().replace(" ", "_") for name in ANALYSIS_CLASSES.values()
}

# display colours (consistent with the study figures)
COLOURS: dict[str, str] = {
    "Private House_A": "#ee7d3f", "Private House_B": "#f5b45f",
    "Carpet Houses": "#c8376e", "Rectangle Building": "#3460c4",
    "Long Rectangle": "#7ea7ea", "H Shape": "#13917f", "U Shape": "#6cc5a4",
    "L Shape": "#2f6b4f", "Geometric Building": "#7b4cc2",
    "Geometric Shape": "#bf9ae8", "Plus Shape": "#7d6b22", UNCLASSIFIED: "#e0e0e0",
}


def raw_to_network(code: int) -> int:
    return RAW_TO_NETWORK.get(int(code), int(code))


def network_name(cid: int) -> str:
    return NETWORK_CLASSES.get(int(cid), UNCLASSIFIED)


def analysis_id(cid: int) -> int:
    return NETWORK_TO_ANALYSIS.get(int(cid), 0)


def analysis_name(cid: int) -> str:
    """Analysis (10-class) name for a NETWORK class id; 0 -> 'None'."""
    a = analysis_id(cid)
    return ANALYSIS_CLASSES.get(a, UNCLASSIFIED)
