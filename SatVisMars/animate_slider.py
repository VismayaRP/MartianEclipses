from __future__ import annotations

from pathlib import Path as FilePath
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.widgets import Slider

from .calc import Visibility

MARS_MAP_PATH = FilePath(__file__).resolve().parent / "img" / "mars_names.png"


class VisibilityWidget:
    """Animate the Phobos visibility footprint over an equirectangular Mars map."""

    def __init__(
        self,
        visibility: Optional[Visibility] = None,
        image_path: FilePath = MARS_MAP_PATH,
        footprint_points: int = 500,
    ):
        self.visibility = visibility or Visibility()
        self.image_path = image_path
        self.footprint_points = footprint_points
        self._fig = None
        self._ax = None
        self._long_slider = None
        self._shade_patch = None
        self._boundary_lines: list = []


    def _setup_figure(self) -> None:
        self._fig, self._ax = plt.subplots(figsize=(12, 6))

        # Set up sliders for longitude and latitude
        self._long_slider_ax = self._fig.add_axes((0.2, 0.01, 0.6, 0.03))
        self._long_slider = Slider(
            self._long_slider_ax,
            "Longitude",
            valmin=-180,
            valmax=180,
            valinit=0,
            valstep=0.1,
            orientation="horizontal",
        )

        # Add slider for radius ratio, see calc.py for more details
        self._radius_ratio_ax = self._fig.add_axes((0.03, 0.2, 0.03, 0.6))
        self._radius_ratio_slider = Slider(
            self._radius_ratio_ax,
            "Sphere Radius Ratio",
            valmin=1e-3,
            valmax=1.0,
            valinit=self.visibility.radius_ratio,
            valstep=1e-5,
            orientation="vertical",
        )
        self._radius_ratio_slider.label.set_position((-0.5, 0.5))
        self._radius_ratio_slider.label.set_rotation(90)
        self._radius_ratio_slider.label.set_ha("center")
        self._radius_ratio_slider.label.set_va("center")

        # Add slider for satellite orbit ratio, see calc.py for more details
        self._satellite_orbit_ratio_ax = self._fig.add_axes((0.92, 0.2, 0.03, 0.6))
        self._satellite_orbit_ratio_slider = Slider(
            self._satellite_orbit_ratio_ax,
            "Satellite Orbit Ratio",
            valmin=1.0,
            valmax=50,
            valinit=self.visibility.satellite_orbit_ratio,
            valstep=0.1,
            orientation="vertical",
        )
        self._satellite_orbit_ratio_slider.label.set_position((1.5, 0.5))
        self._satellite_orbit_ratio_slider.label.set_rotation(-90)
        self._satellite_orbit_ratio_slider.label.set_ha("center")
        self._satellite_orbit_ratio_slider.label.set_va("center")

        map_image = plt.imread(self.image_path)
        self._ax.imshow(
            map_image,
            extent=(-180.0, 180.0, -90.0, 90.0),
            origin="upper",
            aspect="auto",
        )
        self._ax.set_xlim(-180, 180)
        self._ax.set_ylim(-90, 90)
        self._ax.set_xlabel("Longitude (°)")
        self._ax.set_ylabel("Latitude (°)")
        self._ax.set_title("Satellite Visibility Footprint", fontsize=16, pad=12)
        self._ax.grid(linestyle="--", color="white", alpha=0.35)

        self._shade_patch = PathPatch(
            Path([[0, 0]]),
            facecolor="red",
            alpha=0.1,
            linewidth=0,
            edgecolor="none",
        )
        self._ax.add_patch(self._shade_patch)
        (line_east,) = self._ax.plot([], [], color="darkred", linewidth=2)
        (line_west,) = self._ax.plot([], [], color="darkred", linewidth=2)
        self._boundary_lines = [line_east, line_west]

    @staticmethod
    def _world_mask_path(hole_lons: np.ndarray, hole_lats: np.ndarray) -> Path:
        """Build a world-spanning polygon with the visibility footprint cut out."""
        outer = np.array(
            [[-180, -90], [180, -90], [180, 90], [-180, 90], [-180, -90]]
        )
        hole = np.column_stack([hole_lons, hole_lats])
        if not np.allclose(hole[0], hole[-1]):
            hole = np.vstack([hole, hole[0]])

        vertices = np.vstack([outer, hole])
        codes = np.concatenate(
            [
                [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY],
                [Path.MOVETO] + [Path.LINETO] * (len(hole) - 2) + [Path.CLOSEPOLY],
            ]
        )
        return Path(vertices, codes)

    def _update(self, _val: float | None = None) -> None:
        """Redraw footprint from current slider values."""
        self.visibility.update_params(
            self._radius_ratio_slider.val,
            self._satellite_orbit_ratio_slider.val,
        )
        subpoint_lon = self._long_slider.val
        phi_deg, lon_east, lon_west = self.visibility.footprint_boundaries(
            subpoint_lon, self.footprint_points
        )
        poly_lons, poly_lats = self.visibility.footprint_polygon(
            subpoint_lon, self.footprint_points
        )

        self._shade_patch.set_path(self._world_mask_path(poly_lons, poly_lats))
        self._boundary_lines[0].set_data(lon_east, phi_deg)
        self._boundary_lines[1].set_data(lon_west, phi_deg)
        self._fig.canvas.draw_idle()

    def interact(self):
        self._setup_figure()
        self._long_slider.on_changed(self._update)
        self._radius_ratio_slider.on_changed(self._update)
        self._satellite_orbit_ratio_slider.on_changed(self._update)
        self._update()

        plt.show()
