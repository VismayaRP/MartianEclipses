import numpy as np
import cv2
# Distances in 10^3 km
RMars = 3.396
orbit_phobos = 9.4
# Angles in degrees; covers visible range of the moon from the surface
phobvis = np.degrees(np.arccos(RMars/orbit_phobos))

# Pixel width for mars_name_coord.png
width = 2054
# Ratio of radius of visible range to 360 degrees; will be used to convert to pixels
shadratio = phobvis*180 
# 'Radius' of visible range in pixel size; for use to return radius magnitude for use in range mask
radrange = shadratio * width
