import numpy as np
import os
import glob
import random
import argparse
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from src.ParkingEnv import ParkingEnv
import src.Visualization as Visualization
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

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

def run_episode(env, model):
    """Executa um episódio sem gravar vídeo."""
    observation = env.reset()  # VecEnv returns just obs, not (obs, info)
    total_reward = 0.0

    while True:
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)  # VecEnv returns done, not terminated/truncated
        total_reward += float(reward[0])  # reward is an array for VecEnv
        if done[0]:
            break

    return total_reward

def evaluate_model(env, model, iterations=10):
    """Executa vários episódios de avaliação e retorna as recompensas."""
    rewards = []
    for _ in range(iterations):
        rewards.append(run_episode(env, model))
    return rewards

def record_video(env, model, video_name, num_episodes=1):
    """
    Grava um vídeo manualmente usando Visualization.VideoRecorder.
    """
    video_dir = os.path.join(SCRIPT_DIR, "video")
    os.makedirs(video_dir, exist_ok=True)
    
    video_path = os.path.join(video_dir, f"{video_name}.mp4")
    video_recorder = Visualization.VideoRecorder(video_path, fps=10)
    
    total_reward = 0.0
    
    for episode in range(num_episodes):
        observation = env.reset()
        episode_reward = 0.0
        
        while True:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, info = env.step(action)
            episode_reward += float(reward[0])
            # For video, we need to render the underlying env
            video_recorder.append(env.venv.envs[0].render())
            
            if done[0]:
                break
        
        total_reward += episode_reward
    
    video_recorder.close()
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

    # Define path for normalization stats based on the found model
    vec_norm_path = model_path.replace(".zip", "") + "_vecnormalize.pkl"

    # 1. Create a fresh environment (must match the structure of the training env)
    env = DummyVecEnv([lambda: ParkingEnv()])

    # 2. Load the SAVED statistics specifically for this model
    if os.path.exists(vec_norm_path):
        print(f"Loading normalization stats from: {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env)
    else:
        print(f"WARNING: Normalization stats not found at {vec_norm_path}")
        print("Evaluating without loading normalization stats (results may be incorrect if model expected them).")
        # Fallback: wrap in new VecNormalize just to keep API consistent, but it will be uninitialized
        env = VecNormalize(env, training=False, norm_reward=False)

    # 3. CRITICAL: Configure for testing
    env.training = False     # Do not update stats (freeze mean/std)
    env.norm_reward = False  # Return raw rewards (so you see the real score)
    
    model = SAC.load(model_path)
    print("Loaded SAC model.")

    # Pass the normalized env to both functions
    video_filename = os.path.basename(model_path).replace(".zip", "") + "_eval"
    record_video(env, model, video_filename, num_episodes=1)
    
    rewards = evaluate_model(env, model, 10)
    rewards = np.array(rewards)
    
    print(f"Mean reward: {rewards.mean()}")
    print(f"Std reward: {rewards.std()}")
    print(f"Min reward: {rewards.min()}")
    print(f"Max reward: {rewards.max()}")
