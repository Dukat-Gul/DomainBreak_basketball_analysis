import numpy as np


class KalmanFilter:
    """
    Un semplice Filtro di Kalman per tracciare oggetti in 2D.
    """

    def __init__(self):
        self.kf = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32
        )
        self.state = np.array([0, 0, 0, 0], np.float32)
        self.measurement_matrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.process_noise = np.eye(4, dtype=np.float32) * 1e-2
        self.measurement_noise = np.eye(2, dtype=np.float32) * 1e-1
        self.error_covariance = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict(self):
        self.state = np.dot(self.kf, self.state)
        self.error_covariance = (
            np.dot(np.dot(self.kf, self.error_covariance), self.kf.T)
            + self.process_noise
        )
        return self.state[:2]

    def correct(self, measurement):
        measurement = np.array(measurement, dtype=np.float32)
        innovation_covariance = (
            np.dot(
                np.dot(self.measurement_matrix, self.error_covariance),
                self.measurement_matrix.T,
            )
            + self.measurement_noise
        )
        kalman_gain = np.dot(
            np.dot(self.error_covariance, self.measurement_matrix.T),
            np.linalg.inv(innovation_covariance),
        )
        innovation = measurement - np.dot(self.measurement_matrix, self.state)
        self.state = self.state + np.dot(kalman_gain, innovation)
        self.error_covariance = self.error_covariance - np.dot(
            np.dot(kalman_gain, self.measurement_matrix), self.error_covariance
        )
        return self.state[:2]

    def initialize_state(self, measurement):
        measurement = np.array(measurement, dtype=np.float32)
        self.state[:2] = measurement.reshape(2)
        self.initialized = True


# NUOVA CLASSE DA AGGIUNGERE
class DetectionsToTracksKalmanFilter:
    """
    Classe che utilizza il KalmanFilter per associare le detections della palla
    e creare una traccia stabile.
    """

    def __init__(self, max_misses=5):
        self.kf = KalmanFilter()
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

    def process_detections(self, bbox, conf):
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
                pred_center = self.kf.predict()
                # Ricostruisci bbox mantenendo dimensioni precedenti
                new_bbox = self._reconstruct_bbox(pred_center, self.last_bbox)
                self.last_bbox = new_bbox
                return new_bbox
            else:
                return None

        # C'è una detection
        meas_center = self._get_center(bbox)

        if not self.kf.initialized:
            self.kf.initialize_state(meas_center)
            self.misses = 0
            self.last_bbox = bbox
            return bbox

        # Predizione + correzione
        self.kf.predict()
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
