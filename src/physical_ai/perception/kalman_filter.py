import numpy as np


class KalmanFilter2D:
    def __init__(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
    ) -> None:
        self.state = np.array([x, y, vx, vy], dtype=float)

        self.transition_matrix = np.array(
            [
                [1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        self.measurement_matrix = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ],
            dtype=float,
        )

        self.covariance = np.eye(4)
        self.process_noise = np.eye(4) * 0.1
        self.measurement_noise = np.eye(2) * 1.0

    def predict(self) -> np.ndarray:
        self.state = self.transition_matrix @ self.state

        self.covariance = (
            self.transition_matrix
            @ self.covariance
            @ self.transition_matrix.T
            + self.process_noise
        )

        return self.state.copy()

    def update(
        self,
        measured_x: float,
        measured_y: float,
    ) -> np.ndarray:
        measurement = np.array(
            [measured_x, measured_y],
            dtype=float,
        )

        innovation = (
            measurement
            - self.measurement_matrix @ self.state
        )

        innovation_covariance = (
            self.measurement_matrix
            @ self.covariance
            @ self.measurement_matrix.T
            + self.measurement_noise
        )

        kalman_gain = (
            self.covariance
            @ self.measurement_matrix.T
            @ np.linalg.inv(innovation_covariance)
        )

        self.state = (
            self.state
            + kalman_gain @ innovation
        )

        identity = np.eye(4)

        self.covariance = (
            identity
            - kalman_gain @ self.measurement_matrix
        ) @ self.covariance

        return self.state.copy()
