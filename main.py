"""Entry point mirroring the original test script using the refactored modules."""

from animate_slider import VisibilityWidget
from calc import Visibility

if __name__ == "__main__":
    # PhobosVisibilityAnimation().save()
    # Open the file in the default viewer
    # import os
    # os.system(f"open {PhobosVisibilityAnimation().save()}")
    visibility = Visibility(radius_ratio=0.1, satellite_orbit_ratio=2.0)
    widget = VisibilityWidget(visibility=visibility)
    widget.interact()
