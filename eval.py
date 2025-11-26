import numpy as np
import os
import glob
import random
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from src.ParkingEnv import ParkingEnv
import src.Visualization as Visualization

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")
MODEL_NAME = None  # None para carregar o modelo mais recentemente atualizado, ou um nome específico como "SAC_Improved_V1"

def get_latest_model(model_dir):
    """Encontra o arquivo .zip mais recentemente modificado no diretório de modelos."""
    list_of_files = glob.glob(os.path.join(model_dir, "*.zip"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def run_episode(model, seed=None):
    """Executa um episódio sem gravar vídeo."""
    env = ParkingEnv(seed)
    observation, info = env.reset()
    total_reward = 0.0

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break

    env.close()
    return total_reward

def evaluate_model(model, iterations=10):
    """Executa vários episódios de avaliação e retorna as recompensas."""
    rewards = []
    for _ in range(iterations):
        rewards.append(run_episode(model, int(random.random() * 1000)))
    return rewards

def record_video(model, video_name, num_episodes=1):
    """
    Grava um vídeo manualmente usando Visualization.VideoRecorder.
    """
    video_dir = os.path.join(SCRIPT_DIR, "video")
    os.makedirs(video_dir, exist_ok=True)
    
    video_path = os.path.join(video_dir, f"{video_name}.mp4")
    video_recorder = Visualization.VideoRecorder(video_path, fps=10)
    
    env = ParkingEnv()
    total_reward = 0.0
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0.0
        
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            video_recorder.append(env.render())
            
            if terminated or truncated:
                break
        
        total_reward += episode_reward
    
    video_recorder.close()
    env.close()
    print(f"\nVideo recorded with total reward: {total_reward}")
    print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    # Determina o caminho do modelo
    if MODEL_NAME:
        model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME + ".zip")
        if not os.path.exists(model_path):
             # Tenta sem a extensão caso não encontre
            model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME)
            if not os.path.exists(model_path):
                print(f"Model {MODEL_NAME} not found in {MODEL_SAVE_DIR}")
                exit(1)
    else:
        model_path = get_latest_model(MODEL_SAVE_DIR)
        if not model_path:
            print(f"No models found in {MODEL_SAVE_DIR}")
            exit(1)
    
    print(f"Loading model from: {model_path}")
    
    # Load model - try SAC first, then PPO
    try:
        model = SAC.load(model_path)
        print("Loaded SAC model.")
    except Exception:
        try:
            model = PPO.load(model_path)
            print("Loaded PPO model.")
        except Exception as e:
             print(f"Failed to load model as SAC or PPO. Error: {e}")
             exit(1)
             
    # Record one full episode
    video_filename = os.path.basename(model_path).replace(".zip", "") + "_eval"
    record_video(model, video_filename, num_episodes=1)
    
    rewards = evaluate_model(model, 10)
    rewards = np.array(rewards)
    
    print(f"Mean reward: {rewards.mean()}")
    print(f"Std reward: {rewards.std()}")
    print(f"Min reward: {rewards.min()}")
    print(f"Max reward: {rewards.max()}")

