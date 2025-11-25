import shutil
import os
import pathlib

LOG_DIR = str(pathlib.Path(__file__).parent.parent / "logs")
MODEL_SAVE_DIR = str(pathlib.Path(__file__).parent.parent / "models")


def clear_logs_models():
    if os.path.exists(MODEL_SAVE_DIR):
        shutil.rmtree(MODEL_SAVE_DIR)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)
    os.makedirs(LOG_DIR, exist_ok=True)

if __name__ == "__main__":
    clear_logs_models()