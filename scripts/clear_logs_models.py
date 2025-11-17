import shutil
import os

def clear_logs_models():
    if os.path.exists("src/models"):
        shutil.rmtree("src/models")
    os.makedirs("src/models", exist_ok=True)
    if os.path.exists("src/logs"):
        shutil.rmtree("src/logs")
    os.makedirs("src/logs", exist_ok=True)

if __name__ == "__main__":
    clear_logs_models()