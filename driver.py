import carState
import carControl
import joblib
import numpy as np

class Driver:
    def __init__(self, stage):
        # Load model and scalers
        self.model = joblib.load("torcs_ai_model.pkl")
        self.scaler_X = joblib.load("input_scaler.pkl")
        self.scaler_y = joblib.load("output_scaler.pkl")

        # Not used in AI mode but kept for interface compatibility
        self.stage = stage
        self.state = carState.CarState()
        self.control = carControl.CarControl()

    def init(self):
        # Track sensors configuration
        self.angles = [0 for _ in range(19)]
        for i in range(5):
            self.angles[i] = -90 + i * 15
            self.angles[18 - i] = 90 - i * 15
        for i in range(5, 9):
            self.angles[i] = -20 + (i - 5) * 5
            self.angles[18 - i] = 20 - (i - 5) * 5
        return f'({ "init " + " ".join(map(str, self.angles)) })'

    def drive(self, msg):
        self.state.setFromMsg(msg)

        # === Build model input vector ===
        try:
            input_vector = [
                self.state.getSpeedX(),
                self.state.getSpeedY(),
                self.state.getSpeedZ(),
                self.state.getAngle(),
                self.state.getRpm(),
                self.state.getGear(),
                self.state.getTrackPos(),
                *self.state.getTrack(),
                *self.state.getWheelSpinVel()
            ]

            X_scaled = self.scaler_X.transform([input_vector])
            y_scaled = self.model.predict(X_scaled)
            accel, brake, steer = self.scaler_y.inverse_transform(y_scaled)[0]

            # Clamp values
            accel = float(np.clip(accel, 0.0, 1.0))
            brake = float(np.clip(brake, 0.0, 1.0))
            steer = float(np.clip(steer, -1.0, 1.0))

            # Gear shifting logic
            current_rpm = self.state.getRpm()
            current_gear = self.state.getGear()
            speed = self.state.getSpeedX()  # Speed in m/s

            # Simple gear-shifting rules (adjust thresholds as needed)
            if current_gear == 0:  # Neutral, set to 1st gear initially
                current_gear = 1
            else:
                # Shift up if RPM is high and not in max gear (assuming 6 gears)
                if current_rpm > 8000 and current_gear < 6:
                    current_gear += 1
                # Shift down if RPM is too low and not in 1st gear
                elif current_rpm < 3000 and current_gear > 1:
                    current_gear -= 1
                # Emergency downshift if speed is very low to avoid stalling
                if speed < 5.0 and current_gear > 1:
                    current_gear -= 1

            # Set controls
            self.control.setAccel(accel)
            self.control.setBrake(brake)
            self.control.setSteer(steer)
            self.control.setGear(current_gear)

        except Exception as e:
            print(f"[ERROR] AI inference failed: {e}")
            self.control.setAccel(0.0)
            self.control.setBrake(0.0)
            self.control.setSteer(0.0)
            self.control.setGear(1)  # Default to 1st gear on failure

        return self.control.toMsg()

    def onShutDown(self):
        pass

    def onRestart(self):
        pass