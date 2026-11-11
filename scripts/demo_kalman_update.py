from physical_ai.perception.kalman_filter import KalmanFilter2D


def main():
    kalman = KalmanFilter2D(
        x=160,
        y=260,
        vx=10,
        vy=-5,
    )

    print("Initial state:")
    print(kalman.state)
    print()

    prediction = kalman.predict()

    print("Predicted state:")
    print(prediction)
    print()

    updated = kalman.update(
        measured_x=173,
        measured_y=258,
    )

    print("Measurement:")
    print([173, 258])
    print()

    print("Updated state:")
    print(updated)
    print()

    print("Updated covariance:")
    print(kalman.covariance)


if __name__ == "__main__":
    main()
