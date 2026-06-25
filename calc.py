from __future__ import annotations

from typing import Union

import numpy as np

ArrayLike = Union[float, np.ndarray]


class Visibility:
    """
    Visibility region of an spherical satellite on the equirectangular projected 
    sphere host surface using the spherical law of cosines:

        cos(alpha) = cos(theta) * cos(phi - phi0)

    where alpha is the maximum elevation angle (latitude) such that the satellite is visible, 
    theta is surface latitude, phi is surface longitude, and phi0 is the sub-satellite longitude.
    Note that this assumes perfect equitorial orbit, where the satellite is always on top of the equator.
  """
    # Mess around with these values to see how the visibility footprint changes
    def __init__(self, radius_ratio: float = 0.1, satellite_orbit_ratio: float = 2.0):
        """
        Parameters:
            radius_ratio: Ratio of the satellite radius (r) to the host surface radius (R)
            satellite_orbit_ratio: Ratio of the satellite orbit radius (d) to R + r
        """
        self.radius_ratio = radius_ratio
        self.satellite_orbit_ratio = satellite_orbit_ratio
        self._recalculate()

    def _recalculate(self) -> None:
        # Distances in unitless form
        self.r = 1.0
        self.R = self.r / self.radius_ratio
        self.d = self.satellite_orbit_ratio * (self.R + self.r)
        self.k = (self.R - self.r) / self.d
        self.theta_max = np.arccos(self.k)
        self.max_visibility_deg = np.degrees(self.theta_max)

    def update_params(
        self, radius_ratio: float, satellite_orbit_ratio: float
    ) -> None:
        """Recompute geometry when slider values change."""
        self.radius_ratio = radius_ratio
        self.satellite_orbit_ratio = satellite_orbit_ratio
        self._recalculate()

    def _latitude_grid(self, n_points: int = 500) -> np.ndarray:
        return np.linspace(-self.theta_max, self.theta_max, n_points)

    def longitude_offset(self, latitudes_rad: np.ndarray) -> np.ndarray:
        """Angular longitude offset from the sub-satellite point at each latitude."""
        cos_delta = np.clip(self.k / np.cos(latitudes_rad), -1.0, 1.0)
        return np.arccos(cos_delta)

    def footprint_boundaries(
        self, subpoint_longitude_deg: float, n_points: int = 500
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Return latitude (deg) and east/west boundary longitudes (deg) for a
        given Phobos sub-satellite longitude.
        """
        phi = self._latitude_grid(n_points)
        delta_lambda_deg = np.degrees(self.longitude_offset(phi))
        phi_deg = np.degrees(phi)

        lon_east = subpoint_longitude_deg + delta_lambda_deg
        lon_west = subpoint_longitude_deg - delta_lambda_deg
        return phi_deg, lon_east, lon_west

    def footprint_polygon(
        self, subpoint_longitude_deg: float, n_points: int = 500
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return a closed polygon (longitudes, latitudes) tracing the footprint."""
        phi_deg, lon_east, lon_west = self.footprint_boundaries(
            subpoint_longitude_deg, n_points
        )

        west_edge_lons = lon_west
        west_edge_lats = phi_deg
        east_edge_lons = lon_east[::-1]
        east_edge_lats = phi_deg[::-1]

        lons = np.concatenate([west_edge_lons, east_edge_lons])
        lats = np.concatenate([west_edge_lats, east_edge_lats])
        return lons, lats

    @staticmethod
    def lon_lat_to_pixel(
        lon_deg: ArrayLike,
        lat_deg: ArrayLike,
        image_width: int = 2048,
        image_height: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Map geographic coordinates to pixel coordinates on an equirectangular
        image whose center is (0° latitude, 0° longitude).
        """
        lon = np.asarray(lon_deg, dtype=float)
        lat = np.asarray(lat_deg, dtype=float)
        x = (lon + 180.0) / 360.0 * image_width
        y = (90.0 - lat) / 180.0 * image_height
        return x, y

    @staticmethod
    def pixel_to_lon_lat(
        x: ArrayLike,
        y: ArrayLike,
        image_width: int = 2048,
        image_height: int = 1024,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Inverse of lon_lat_to_pixel for equirectangular images."""
        px = np.asarray(x, dtype=float)
        py = np.asarray(y, dtype=float)
        lon = px / image_width * 360.0 - 180.0
        lat = 90.0 - py / image_height * 180.0
        return lon, lat
