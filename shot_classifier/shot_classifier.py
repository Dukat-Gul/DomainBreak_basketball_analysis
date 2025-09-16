class ShotClassifier:
    def __init__(self, court_dimensions, three_pt_radius_m: float = 6.75):
        self.court_dimensions = court_dimensions
        # Raggio linea da 3 punti in metri (FIBA)
        self.three_pt_radius_m = float(three_pt_radius_m)

    def classify(self, shot_event, player_position):
        """
        Fallback semplice basato sulla coordinata X del bbox (storico).
        Conservato per retrocompatibilità quando non sono disponibili posizioni tattiche.
        """
        x = (player_position[0] + player_position[2]) / 2
        shot_type = "2-pointer"
        # Soglia fissa storica: 250 px
        if x < 250:
            shot_type = "3-pointer"
        return {"shot_type": shot_type}

    def classify_tactical(self, player_tactical_xy_px, tactical_mapper) -> dict:
        """
        Classifica il tiro usando la distanza dal canestro più vicino nella vista tattica.

        - Converte la posizione tattica in metri.
        - Calcola la minima distanza ai due canestri (sinistra/destra) in metri.
        - Se distanza >= 6.75m => 3-pointer, altrimenti 2-pointer.
        """
        if player_tactical_xy_px is None:
            return {"shot_type": "unknown"}

        x_px, y_px = float(player_tactical_xy_px[0]), float(player_tactical_xy_px[1])
        x_m, y_m = tactical_mapper.tactical_px_to_m(x_px, y_px)

        bL, bR = tactical_mapper.basket_centers_m()
        import math

        dL = math.hypot(x_m - bL[0], y_m - bL[1])
        dR = math.hypot(x_m - bR[0], y_m - bR[1])
        d = min(dL, dR)

        shot_type = "3-pointer" if d >= self.three_pt_radius_m else "2-pointer"
        return {"shot_type": shot_type, "distance_m": d}
