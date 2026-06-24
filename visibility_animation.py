from __future__ import annotations

from pathlib import Path as FilePath
from typing import Optional

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from calculations import PhobosVisibility

MARS_MAP_PATH = FilePath(__file__).resolve().parent / "img" / "mars_names.png"
# PHOBOS_IMAGE_PATH = FilePath(__file__).resolve().parent / "img" / "Phobos.png"
DEFAULT_OUTPUT = "phobos_visibility.mp4"


class PhobosVisibilityAnimation:
    """Animate the Phobos visibility footprint over an equirectangular Mars map."""

    def __init__(
        self,
        visibility: Optional[PhobosVisibility] = None,
        image_path: FilePath = MARS_MAP_PATH,
        frames: int = 120,
        start_longitude: float = -180.0,
        end_longitude: float = 180.0,
        footprint_points: int = 500,
        # phobos_image_path: FilePath = PHOBOS_IMAGE_PATH,
        phobos_size_deg: float = 7.0,
        eclipse_longitude: float = 0.0,
        eclipse_half_width_deg: float = 22.0,
    ):
        self.visibility = visibility or PhobosVisibility()
        self.image_path = FilePath(image_path)
        # self.phobos_image_path = FilePath(phobos_image_path)
        self.frames = frames
        self.start_longitude = start_longitude
        self.end_longitude = end_longitude
        self.footprint_points = footprint_points
        self.phobos_size_deg = phobos_size_deg
        self.eclipse_longitude = eclipse_longitude
        self.eclipse_half_width_deg = eclipse_half_width_deg

        self._fig = None
        self._ax = None
        self._shade_patch = None
        self._boundary_lines: list = []
        self._phobos_artist = None
        self._phobos_rgba: Optional[np.ndarray] = None

    def _subpoint_longitude(self, frame: int) -> float:
        """Return the sub-satellite longitude for the given frame."""
        t = frame / self.frames
        return self.start_longitude + (self.end_longitude - self.start_longitude) * t

    @staticmethod
    def _load_phobos_rgba(image_path: FilePath) -> np.ndarray:
        """Load Phobos with a transparent background (black pixels become alpha=0)."""
        rgba = plt.imread(image_path)
        if rgba.ndim == 2:
            rgba = np.stack([rgba, rgba, rgba, np.ones_like(rgba)], axis=-1)
        elif rgba.shape[-1] == 3:
            brightness = rgba.sum(axis=-1)
            alpha = np.where(brightness < 0.08, 0.0, 1.0)
            rgba = np.dstack([rgba, alpha])
        return rgba.astype(float)

    def _phobos_with_eclipse_shadow(self, subpoint_longitude: float) -> np.ndarray:
        """
        Darken Phobos where Mars' umbra falls. Eclipse is maximal when the
        sub-satellite point crosses eclipse_longitude (0° by default).
        """
        frame = self._phobos_rgba.copy()
        height, width = frame.shape[:2]
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        radius = min(center_x, center_y) * 0.92

        yy, xx = np.mgrid[0:height, 0:width]
        norm_x = (xx - center_x) / radius
        norm_y = (yy - center_y) / radius
        on_disk = norm_x**2 + norm_y**2 <= 1.0

        delta_lon = subpoint_longitude - self.eclipse_longitude
        half_width = self.eclipse_half_width_deg
        if abs(delta_lon) >= half_width:
            return frame

        coverage = 1.0 - abs(delta_lon) / half_width
        # Shadow sweeps east-to-west as the sub-point approaches and passes 0° longitude.
        shadow_edge_x = 1.0 - 2.0 * coverage
        penumbra = 0.12
        darken = np.clip((norm_x - shadow_edge_x) / penumbra, 0.0, 1.0) * 0.82
        darken *= on_disk

        for channel in range(3):
            frame[:, :, channel] = np.clip(
                frame[:, :, channel] * (1.0 - darken), 0.0, 1.0
            )
        return frame

    def _phobos_extent(self) -> tuple[float, float, float, float]:
        half = self.phobos_size_deg / 2.0
        return (-half, half, -half, half)

    def _setup_figure(self) -> None:
        self._fig, self._ax = plt.subplots(figsize=(12, 6))

        map_image = plt.imread(self.image_path)
        self._ax.imshow(
            map_image,
            extent=[-180, 180, -90, 90],
            origin="upper",
            aspect="auto",
        )
        self._ax.set_xlim(-180, 180)
        self._ax.set_ylim(-90, 90)
        self._ax.set_xlabel("Longitude (°)")
        self._ax.set_ylabel("Latitude (°)")
        self._ax.set_title("Phobos Visibility Footprint", fontsize=16, pad=12)
        self._ax.grid(linestyle="--", color="white", alpha=0.35)

        self._shade_patch = PathPatch(
            Path([[0, 0]]),
            facecolor="red",
            alpha=0.4,
            linewidth=0,
            edgecolor="none",
        )
        self._ax.add_patch(self._shade_patch)
        (line_east,) = self._ax.plot([], [], color="darkred", linewidth=2)
        (line_west,) = self._ax.plot([], [], color="darkred", linewidth=2)
        self._boundary_lines = [line_east, line_west]

        # self._phobos_rgba = self._load_phobos_rgba(self.phobos_image_path)
        # initial_frame = self._phobos_with_eclipse_shadow(self.start_longitude)
        # self._phobos_artist = self._ax.imshow(
        #     initial_frame,
        #     extent=self._phobos_extent(),
        #     origin="upper",
        #     aspect="equal",
        #     zorder=5,
        #     interpolation="bilinear",
        # )

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

    def _update(self, frame: int):
        subpoint_lon = self._subpoint_longitude(frame)
        phi_deg, lon_east, lon_west = self.visibility.footprint_boundaries(
            subpoint_lon, self.footprint_points
        )
        poly_lons, poly_lats = self.visibility.footprint_polygon(
            subpoint_lon, self.footprint_points
        )

        self._shade_patch.set_path(self._world_mask_path(poly_lons, poly_lats))
        self._boundary_lines[0].set_data(lon_east, phi_deg)
        self._boundary_lines[1].set_data(lon_west, phi_deg)
        # self._phobos_artist.set_data(
        #     self._phobos_with_eclipse_shadow(subpoint_lon)
        # )

        artists = [self._shade_patch, *self._boundary_lines, self._phobos_artist]
        return artists

    def build(self) -> animation.FuncAnimation:
        self._setup_figure()
        return animation.FuncAnimation(
            self._fig,
            self._update,
            frames=self.frames,
            interval=50,
            blit=False,
        )

    def save(
        self,
        output_path: FilePath = DEFAULT_OUTPUT,
        fps: int = 30,
        bitrate: int = 1800,
    ) -> FilePath:
        output_path = FilePath(output_path)
        ani = self.build()
        writer = animation.FFMpegWriter(
            fps=fps,
            metadata={"artist": "MartianEclipses"},
            bitrate=bitrate,
        )
        print(f"Rendering animation to {output_path}...")
        ani.save(str(output_path), writer=writer)
        print(f"Saved to {output_path}")
        return output_path


if __name__ == "__main__":
    PhobosVisibilityAnimation().save()
