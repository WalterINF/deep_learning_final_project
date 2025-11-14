#!/usr/bin/env python3
import subprocess
import sys
import os

# --- Configuration ---
LOG_DIR = "src\logs"
# ---------------------

def launch_tensorboard():
    """
    Launches the TensorBoard server.
    """
    # Ensure the log directory exists, or TensorBoard might complain
    if not os.path.exists(LOG_DIR):
        print(f"Creating log directory: {LOG_DIR}")
        os.makedirs(LOG_DIR)

    # Construct the command
    command = ["tensorboard", "--logdir", LOG_DIR]

    print("-" * 60)
    print(f"Attempting to launch TensorBoard...")
    print(f"Log Directory: {LOG_DIR}")
    print(f"Command:       {' '.join(command)}")
    print("\nView in your browser at: http://localhost:6006/")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("-" * 60)

    try:
        # Execute the command. 
        # This script will block here until you stop TensorBoard (e.g., Ctrl+C).
        subprocess.run(command, check=True)

    except FileNotFoundError:
        print("\n[Error] 'tensorboard' command not found.", file=sys.stderr)
        print("Please ensure TensorBoard is installed and in your system's PATH.", file=sys.stderr)
        print("Try: pip install tensorboard", file=sys.stderr)
    
    except KeyboardInterrupt:
        # Handle the user pressing Ctrl+C
        print("\nTensorBoard server manually shut down.")
    
    except Exception as e:
        # Catch other potential errors
        print(f"\nAn error occurred: {e}", file=sys.stderr)

if __name__ == "__main__":
    launch_tensorboard()