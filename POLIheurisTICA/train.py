import numpy as np
import gymnasium as gym
import stable_baselines3
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import torch
import os
import platform
import sys

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project root is on sys.path so 'src' (namespace package) resolves when running this file directly
PROJECT_ROOT = SCRIPT_DIR  # train.py is at project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Support both running as a module (python -m src.train) and as a script (python src/train.py)
try:
    from ParkingEnv import ParkingEnv  # when importing via project root
except ImportError:
    from ParkingEnv import ParkingEnv  # when running as a script

# config do treinamento - paths relative to script location
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")  # diretório para logs do TensorBoard
MODEL_SAVE_DIR = os.path.join(SCRIPT_DIR, "models")  # diretório para salvar os modelos
MODEL_NAME = "SAC_Improved_V6" # nome do modelo para carregar/treinar/salvar (V2 with improved obs/rewards)
TOTAL_TIMESTEPS = 20_000_000 # total de timesteps para treinar
SAVE_EVERY = 100_000 # salvar o modelo a cada 100.000 timesteps

def make_env(seed: int = 0):
    def _init():
        env = ParkingEnv()
        env = Monitor(env, LOG_DIR)
        env.reset(seed=seed)
        return env
    return _init

def make_vector_env(n_envs: int = 8, use_subproc: bool | None = None):
    if use_subproc is None:
        use_subproc = platform.system() != "Windows"

    env_fns = [make_env(seed=i) for i in range(n_envs)]

    if use_subproc:
        try:
            return SubprocVecEnv(env_fns)
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv.")

    return DummyVecEnv(env_fns)

def train_sac(model: SAC | None = None, total_timesteps: int = 10000, save_every: int | None = None, save_path: str | None = None, save_name: str | None = None):
    env = make_vector_env()

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[512, 256, 256], qf=[512, 256, 256])
    )

    if model is None:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=LOG_DIR,
            learning_rate=3e-4,           
            buffer_size=1_000_000,        
            batch_size=512,               
            gamma=0.99,                   
            ent_coef="auto",              
            target_entropy="auto",        
            tau=0.005,                    
            learning_starts=10_000,       
            train_freq=1,                 
            gradient_steps=1,             
            device="auto",
            use_sde=True,                 
            sde_sample_freq=4,            
        )
    else:
        model.set_env(env)

    timesteps_split = 0
    if save_every is not None:
        timesteps_split = int(total_timesteps / save_every)
        for i in range(timesteps_split):
            model.learn(total_timesteps=save_every, progress_bar=True, tb_log_name=save_name, reset_num_timesteps=False)
            
            if save_path and save_name:
                model_save_path = os.path.join(save_path, save_name)
                model.save(model_save_path)
                print(f"modelo salvo em {model_save_path}")

    return model

if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    print(f"Começando treinamento com modelo {MODEL_NAME}...")
    
    # carrega modelo existente se disponível
    model_path = os.path.join(MODEL_SAVE_DIR, MODEL_NAME + ".zip")
    model = None
    if os.path.exists(model_path):
        print(f"Carregando modelo existente de {model_path}")
        model = SAC.load(model_path)
    
    train_sac(model, total_timesteps=TOTAL_TIMESTEPS, save_every=SAVE_EVERY, save_path=MODEL_SAVE_DIR, save_name=MODEL_NAME)

