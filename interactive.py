#!/usr/bin/env python3
"""
Interactive Parking Environment Controller

Use arrow keys to control the vehicle:
  UP    - Accelerate forward
  DOWN  - Accelerate backward (reverse)
  LEFT  - Steer left
  RIGHT - Steer right
  
  R     - Reset environment
  Q/ESC - Quit

The environment updates automatically every dt seconds.
"""

import cv2
import numpy as np
import time
from src.ParkingEnv import ParkingEnv


def main():
    # Create environment
    env = ParkingEnv(heuristica="euclidiana")
    obs, info = env.reset()
    
    # Control parameters
    velocity = 0.0
    steering = 0.0
    velocity_step = 1.0  # m/s per keypress
    steering_step = np.deg2rad(5.0)  # radians per keypress
    
    # Limits from environment
    max_velocity = env.SPEED_LIMIT_MS
    max_steering = env.STEERING_LIMIT_RAD
    dt = env.DT  # Simulation timestep in seconds
    
    print(__doc__)
    print(f"Velocity range: [{-max_velocity:.1f}, {max_velocity:.1f}] m/s")
    print(f"Steering range: [{-np.rad2deg(max_steering):.1f}, {np.rad2deg(max_steering):.1f}] degrees")
    print(f"Update interval: {dt:.2f} seconds")
    print()
    
    # Initial render
    frame = env.render()
    # Convert list[list[list[int]]] to numpy array
    frame = np.array(frame, dtype=np.uint8)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Scale up for better visibility
    scale = 2
    frame_scaled = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow("Parking Environment", frame_scaled)
    
    total_reward = 0.0
    step_count = 0
    last_update_time = time.time()
    episode_ended = False
    
    while True:
        dt_ms = int(dt * 1000)
        key = cv2.waitKey(dt_ms) & 0xFF
        
        if key != 255:  # A key was pressed
            if key == ord('q') or key == 27:  # 'q' or ESC
                print("\nQuitting...")
                break
            elif key == ord('r'):  # Reset
                obs, info = env.reset()
                velocity = 0.0
                steering = 0.0
                total_reward = 0.0
                step_count = 0
                episode_ended = False
                last_update_time = time.time()
                print("\n--- Environment Reset ---")
            elif key == 82 or key == ord('w'):  # UP arrow or 'w'
                velocity = min(velocity + velocity_step, max_velocity)
            elif key == 84 or key == ord('s'):  # DOWN arrow or 's'
                velocity = max(velocity - velocity_step, -max_velocity)
            elif key == 81 or key == ord('a'):  # LEFT arrow or 'a'
                steering = min(steering - steering_step, max_steering)
            elif key == 83 or key == ord('d'):  # RIGHT arrow or 'd'
                steering = max(steering + steering_step, -max_steering)
            elif key == ord(' '):  # Space to brake/stop
                velocity = 0.0
                steering = 0.0
        
        # Update environment every dt seconds
        current_time = time.time()
        if current_time - last_update_time >= dt and not episode_ended:
            # Take a step in the environment
            action = np.array([velocity, steering], dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            last_update_time = current_time
            
            # Print status
            status = f"Step {step_count:4d} | Vel: {velocity:+5.1f} m/s | Steer: {np.rad2deg(steering):+6.1f}° | Reward: {reward:+7.2f} | Total: {total_reward:+8.2f}"
            print(f"\r{status}", end="", flush=True)
            
            if terminated or truncated:
                episode_ended = True
                if info.get("is_success", False):
                    print(f"\n\n🎉 SUCCESS! Parked successfully!")
                else:
                    print(f"\n\n💥 Episode ended (collision/jackknife/timeout)")
                print(f"Total steps: {step_count}, Total reward: {total_reward:.2f}")
                print("Press 'R' to reset or 'Q' to quit")
        
        # Render current state (always render, even if episode ended)
        frame = env.render()
        # Convert list[list[list[int]]] to numpy array
        frame = np.array(frame, dtype=np.uint8)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Add HUD overlay
        hud_frame = frame_bgr.copy()
        cv2.putText(hud_frame, f"V: {velocity:+.1f}", (5, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(hud_frame, f"S: {np.rad2deg(steering):+.0f}", (5, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(hud_frame, f"R: {total_reward:+.1f}", (5, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        if episode_ended:
            cv2.putText(hud_frame, "EPISODE ENDED", (5, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Scale up for better visibility
        frame_scaled = cv2.resize(hud_frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        
        cv2.imshow("Parking Environment", frame_scaled)
    
    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()

