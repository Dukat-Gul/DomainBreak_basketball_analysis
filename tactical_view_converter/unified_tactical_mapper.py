import os
import sys
from typing import List, Dict, Optional

# Import the two existing implementations with aliases to avoid name clashes
from tactical_view.tactical_view_converter import (
    TacticalViewConverter as WorldToImageConverter,
)
from tactical_view_converter.tactical_view_converter import (
    TacticalViewConverter as ImageToFieldConverter,
)


class UnifiedTacticalMapper:
    """
    Facade that unifies tactical view utilities:
    - Projects 3D world points (court/basket) to image coordinates (WorldToImageConverter)
    - Validates court keypoints and maps player positions to tactical view (ImageToFieldConverter)

    Exposes a consistent API to the main pipeline while keeping existing modules intact.
    """

    def __init__(self):
        # Converter used to project 3D court/basket coordinates into image 2D
        self._w2i = WorldToImageConverter()
        # Converter used for image<->tactical (homography) and player mapping
        # It requires a court image path in its constructor; provide a placeholder image
        # but we only use its homography/utilities, not the image drawing in this facade.
        # Supply a default court image path that exists in repo or caller can override.
        # Here we use a blank path since we won't call its image I/O methods.
        self._i2f = ImageToFieldConverter(court_image_path="")

    # --- Properties bridging underlying converters ---
    @property
    def basket_3d_coordinates(self):
        return self._w2i.basket_3d_coordinates

    @property
    def width_px(self) -> int:
        return self._i2f.width

    @property
    def height_px(self) -> int:
        return self._i2f.height

    @property
    def width_m(self) -> float:
        return self._i2f.actual_width_in_meters

    @property
    def height_m(self) -> float:
        return self._i2f.actual_height_in_meters

    # --- Basket/world projection (compatibility with previous API) ---
    def transform_3d_to_2d(self, points_3d, court_keypoints_for_frame):
        """Delegate to the original world->image converter for basket projection."""
        return self._w2i.transform_3d_to_2d(points_3d, court_keypoints_for_frame)

    # --- Keypoints validation and player mapping to tactical view ---
    def validate_keypoints(self, keypoints_list):
        return self._i2f.validate_keypoints(keypoints_list)

    def transform_players_to_tactical_view(
        self,
        keypoints_list,
        player_tracks: List[Dict[int, Dict]],
    ) -> List[Dict[int, List[float]]]:
        return self._i2f.transform_players_to_tactical_view(keypoints_list, player_tracks)

    # --- Utility: convert tactical px to meters and vice versa ---
    def tactical_px_to_m(self, x_px: float, y_px: float) -> (float, float):
        xm = float(x_px) * (self.width_m / float(self.width_px))
        ym = float(y_px) * (self.height_m / float(self.height_px))
        return xm, ym

    def tactical_m_to_px(self, x_m: float, y_m: float) -> (float, float):
        xp = float(x_m) * (float(self.width_px) / self.width_m)
        yp = float(y_m) * (float(self.height_px) / self.height_m)
        return xp, yp

    def basket_centers_m(self):
        """
        Returns both basket centers in meters in the tactical field coordinate system.
        Left basket at y=0, right basket at y=height_m, both centered at x=width_m/2.
        """
        x = self.width_m / 2.0
        return (x, 0.0), (x, self.height_m)

