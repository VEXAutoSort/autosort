"""Classification camera — name a VEX piece by measuring it on the white stage.

    frame - stage reference  ->  mask  ->  mm measurement  ->  nearest catalog entry
"""
from .camera import BoxCamera
from .catalog import Catalog, Entry, Match
from .classifier import UNKNOWN, Classifier
from .vision import Shape, measure, outline, segment

__all__ = [
    "BoxCamera", "Catalog", "Entry", "Match",
    "Classifier", "UNKNOWN",
    "Shape", "measure", "outline", "segment",
]
