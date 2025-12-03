import numpy as np
import os
import glob
import random
import argparse
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from src.ParkingEnv import ParkingEnv
import src.Visualization as Visualization

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")

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
    parser = argparse.ArgumentParser(description="Evaluate a trained parking model.")
    parser.add_argument(
        "model_name",
        nargs="?",
        default=None,
        help="Name of the model to load (without .zip extension). If not provided, loads the most recently updated model."
    )
    args = parser.parse_args()

    # Determina o caminho do modelo
    if args.model_name:
        model_path = os.path.join(MODEL_SAVE_DIR, args.model_name + ".zip")
        if not os.path.exists(model_path):
             # Tenta sem a extensão caso não encontre
            model_path = os.path.join(MODEL_SAVE_DIR, args.model_name)
            if not os.path.exists(model_path):
                print(f"Model {args.model_name} not found in {MODEL_SAVE_DIR}")
                exit(1)
    else:
        model_path = get_latest_model(MODEL_SAVE_DIR)
        if not model_path:
            print(f"No models found in {MODEL_SAVE_DIR}")
            exit(1)
    
    print(f"Loading model from: {model_path}")
    
    model = SAC.load(model_path)
    print("Loaded SAC model.")
             
    # grava episodio
    video_filename = os.path.basename(model_path).replace(".zip", "") + "_eval"
    record_video(model, video_filename, num_episodes=1)
    
    rewards = evaluate_model(model, 10)
    rewards = np.array(rewards)
    
    print(f"Mean reward: {rewards.mean()}")
    print(f"Std reward: {rewards.std()}")
    print(f"Min reward: {rewards.min()}")
    print(f"Max reward: {rewards.max()}")

