# Copyright (c) 2026 Omair Kamil
# See LICENSE file in root directory for license terms.

import numpy as np
from olive_solve import FusedSolver
import time

def main():
    print("Initializing FusedSolver...")
    # Assume we don't need a real db just to test the API bounds
    solver = FusedSolver("./tetra3/tests/fixtures/default_database.npz")
    
    print("Starting IMU...")
    try:
        success = solver.start_imu()
        print(f"IMU start result: {success}")
    except Exception as e:
        print(f"IMU start threw exception: {e}")
        
    print("Testing get_sensor_data()...")
    time.sleep(0.5)
    try:
        data = solver.get_sensor_data()
        if data is None:
            print("No sensor data available.")
        else:
            print("Sensor Data:")
            for k, v in data.items():
                print(f"  {k}: {v}")
    except Exception as e:
        print(f"get_sensor_data error: {e}")
        
    print("Stopping IMU...")
    solver.stop_imu()
    print("Done!")

if __name__ == "__main__":
    main()
