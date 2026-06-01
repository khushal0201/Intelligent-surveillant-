"""Polygon zones and entry-line crossing utilities (normalised coordinates)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from shapely.geometry import Polygon, Point


@dataclass
class EntryLine:
    p1: tuple[float, float]
    p2: tuple[float, float]
    inbound_normal: tuple[float, float]  # unit-ish vector pointing INTO the store

    def side(self, x: float, y: float) -> float:
        """Signed projection onto inbound_normal relative to line midpoint.
        Positive => inside / inbound side; Negative => outside.
        """
        mx = (self.p1[0] + self.p2[0]) / 2.0
        my = (self.p1[1] + self.p2[1]) / 2.0
        nx, ny = self.inbound_normal
        return (x - mx) * nx + (y - my) * ny


class ZoneSet:
    """Holds named polygons + optional entry line for a single camera."""

    def __init__(self, zones: dict[str, list[list[float]]],
                 entry_line: Optional[dict] = None):
        self.polys: dict[str, Polygon] = {
            name: Polygon(pts) for name, pts in zones.items()
        }
        self.entry_line: Optional[EntryLine] = None
        if entry_line:
            self.entry_line = EntryLine(
                p1=tuple(entry_line["p1"]),
                p2=tuple(entry_line["p2"]),
                inbound_normal=tuple(entry_line["inbound_normal"]),
            )

    def zone_at(self, x_norm: float, y_norm: float) -> Optional[str]:
        p = Point(x_norm, y_norm)
        for name, poly in self.polys.items():
            if poly.contains(p):
                return name
        return None
