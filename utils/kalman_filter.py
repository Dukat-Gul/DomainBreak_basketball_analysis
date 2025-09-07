import numpy as np


class KalmanFilter:
    """
    Filtro di Kalman a accelerazione costante (CA) per tracking 2D.

    Stato: x = [x, y, vx, vy, ax, ay]^T
    Misura: z = [x, y]^T
    """

    def __init__(self, meas_std: float = 5.0, acc_std: float = 1.5):
        # Stato e matrici principali
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.P = np.eye(6, dtype=np.float32) * 1e2  # incertezza iniziale ampia
        self.F = np.eye(6, dtype=np.float32)
        self.H = np.zeros((2, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Rumori
        self.meas_std = float(meas_std)
        self.R = np.eye(2, dtype=np.float32) * (self.meas_std ** 2)
        self.acc_std = float(acc_std)
        self.Q = np.eye(6, dtype=np.float32)

        self.initialized = False

    def _update_F_Q(self, dt: float):
        dt = float(dt)
        # Matrice di transizione per modello CA
        self.F[...] = np.eye(6, dtype=np.float32)
        self.F[0, 2] = dt
        self.F[1, 3] = dt
        self.F[0, 4] = 0.5 * dt * dt
        self.F[1, 5] = 0.5 * dt * dt
        self.F[2, 4] = dt
        self.F[3, 5] = dt

        # Rumore di processo: assumiamo rumore su accelerazione (jerk non modellato)
        # Impostiamo Q con pesi maggiori sui termini di accelerazione e velocità
        q_pos = (0.25 * dt ** 4) * (self.acc_std ** 2)
        q_vel = (0.5 * dt ** 2) * (self.acc_std ** 2)
        q_acc = (self.acc_std ** 2)
        self.Q[...] = 0.0
        self.Q[0, 0] = q_pos
        self.Q[1, 1] = q_pos
        self.Q[2, 2] = q_vel
        self.Q[3, 3] = q_vel
        self.Q[4, 4] = q_acc
        self.Q[5, 5] = q_acc

    def initialize_state(self, measurement, dt: float = 1.0):
        z = np.array(measurement, dtype=np.float32).reshape(2, 1)
        self._update_F_Q(dt)
        self.x[...] = 0.0
        self.x[0:2, 0] = z[:, 0]
        # Inizializza P con incertezze ragionevoli: pos basso, vel/acc alti
        self.P = np.diag([10.0, 10.0, 100.0, 100.0, 400.0, 400.0]).astype(np.float32)
        self.initialized = True

    def predict(self, dt: float = 1.0):
        if not self.initialized:
            return None
        self._update_F_Q(dt)
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # Ritorna la misura prevista (x,y)
        z_pred = self.H @ self.x
        return z_pred.flatten()

    def correct(self, measurement):
        if not self.initialized:
            return None
        z = np.array(measurement, dtype=np.float32).reshape(2, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - (self.H @ self.x)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P
        z_corr = self.H @ self.x
        return z_corr.flatten()

    def peek(self, dt: float = 1.0):
        """
        Calcola (senza aggiornare lo stato interno) la proiezione della misura
        e la matrice di innovazione S per un passo futuro con passo dt.
        Utile per il gating di Mahalanobis.
        """
        if not self.initialized:
            return None, None
        # Copie locali
        x = self.x.copy()
        P = self.P.copy()
        # Aggiorna F e Q per dt
        self._update_F_Q(dt)
        x_pred = self.F @ x
        P_pred = self.F @ P @ self.F.T + self.Q
        z_pred = self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        return z_pred.flatten(), S


# NUOVA CLASSE DA AGGIUNGERE
class DetectionsToTracksKalmanFilter:
    """
    Classe che utilizza il KalmanFilter per associare le detections della palla
    e creare una traccia stabile.
    """

    def __init__(self, max_misses=5, meas_std: float = 5.0, acc_std: float = 1.5):
        self.kf = KalmanFilter(meas_std=meas_std, acc_std=acc_std)
        self.max_misses = max_misses
        self.misses = 0
        self.last_bbox = None

    def _get_center(self, bbox):
        return [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]

    def _reconstruct_bbox(self, center, original_bbox):
        w = original_bbox[2] - original_bbox[0]
        h = original_bbox[3] - original_bbox[1]
        return [
            center[0] - w / 2,
            center[1] - h / 2,
            center[0] + w / 2,
            center[1] + h / 2,
        ]

    def process_detections(self, bbox, conf, dt: float = 1.0):
        """
        Aggiorna la traccia dato un bbox di detection (o nessuna) e ritorna
        il bbox stimato/corretto. Mantiene la traccia viva fino a max_misses
        usando la sola predizione del filtro di Kalman.
        """
        if not bbox:
            if self.kf.initialized and self.last_bbox is not None:
                self.misses += 1
                if self.misses > self.max_misses:
                    self.kf.initialized = False
                    self.misses = 0
                    self.last_bbox = None
                    return None

                # Prevedi la prossima posizione se la detection è mancata
                pred_center = self.kf.predict(dt)
                # Ricostruisci bbox mantenendo dimensioni precedenti
                new_bbox = self._reconstruct_bbox(pred_center, self.last_bbox)
                self.last_bbox = new_bbox
                return new_bbox
            else:
                return None

        # C'è una detection
        meas_center = self._get_center(bbox)

        if not self.kf.initialized:
            self.kf.initialize_state(meas_center, dt)
            self.misses = 0
            self.last_bbox = bbox
            return bbox

        # Predizione + correzione
        self.kf.predict(dt)
        corrected_center = self.kf.correct(meas_center)

        # Smussa le dimensioni per evitare salti improvvisi
        if self.last_bbox is not None:
            prev_w = self.last_bbox[2] - self.last_bbox[0]
            prev_h = self.last_bbox[3] - self.last_bbox[1]
        else:
            prev_w = bbox[2] - bbox[0]
            prev_h = bbox[3] - bbox[1]

        meas_w = bbox[2] - bbox[0]
        meas_h = bbox[3] - bbox[1]
        new_w = 0.7 * prev_w + 0.3 * meas_w
        new_h = 0.7 * prev_h + 0.3 * meas_h

        new_bbox = [
            float(corrected_center[0] - new_w / 2.0),
            float(corrected_center[1] - new_h / 2.0),
            float(corrected_center[0] + new_w / 2.0),
            float(corrected_center[1] + new_h / 2.0),
        ]

        self.last_bbox = new_bbox
        self.misses = 0
        return new_bbox
