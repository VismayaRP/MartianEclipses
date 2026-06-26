from .animate_slider import VisibilityWidget
from .calc import Visibility


def main() -> None:
    visibility = Visibility(radius_ratio=0.1, satellite_orbit_ratio=2.0)
    widget = VisibilityWidget(visibility=visibility)
    widget.interact()


if __name__ == "__main__":
    main()
