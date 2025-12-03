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
import argparse
from datetime import datetime
from stable_baselines3.common.vec_env import VecNormalize


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure project root is on sys.path so 'src' (namespace package) resolves when running this file directly
PROJECT_ROOT = SCRIPT_DIR  # train.py is at project root
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Support both running as a module (python -m src.train) and as a script (python src/train.py)
try:
    from src.ParkingEnv import ParkingEnv  # when importing via project root
except ModuleNotFoundError:
    from src.ParkingEnv import ParkingEnv  # when running from within src/

# Heurísticas disponíveis
HEURISTICS = ["nao_holonomica", "euclidiana", "manhattan", "nenhuma"]

def get_physical_cores():
    """Retorna o número de núcleos físicos (não threads) do sistema."""
    try:
        import psutil
        return psutil.cpu_count(logical=False) or 1
    except ImportError:
        # Fallback: assume hyperthreading (2 threads por core)
        logical_cores = os.cpu_count() or 2
        return max(1, logical_cores // 2)

def parse_args():
    """Parse command-line arguments."""
    default_n_envs = max(1, get_physical_cores() - 2)
    default_heuristic = "nao_holonomica"
    
    parser = argparse.ArgumentParser(
        description="Treinar modelo SAC para estacionamento autônomo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--modelo", "-m",
        type=str,
        default=None,
        help="Nome do modelo para criar/carregar. Padrão: SAC_<data>-<hora>-<heuristica>"
    )
    
    parser.add_argument(
        "--heuristica", "-H",
        type=str,
        choices=HEURISTICS,
        default=default_heuristic,
        help=f"Heurística usada: {', '.join(HEURISTICS)}"
    )
    
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=10_000_000,
        help="Total de timesteps para treinar"
    )
    
    parser.add_argument(
        "--salvar-cada", "-s",
        type=int,
        default=100_000,
        help="Salvar o modelo a cada X timesteps"
    )
    
    parser.add_argument(
        "--n-envs", "-n",
        type=int,
        default=default_n_envs,
        help=f"Número de ambientes paralelos (padrão: núcleos físicos - 2 = {default_n_envs})"
    )
    
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        default=os.path.join(SCRIPT_DIR, "logs"),
        help="Diretório para logs do TensorBoard"
    )
    
    parser.add_argument(
        "--model-dir", "-d",
        type=str,
        default=os.path.join(SCRIPT_DIR, "models"),
        help="Diretório para salvar os modelos"
    )
    
    args = parser.parse_args()
    
    # Gerar nome do modelo se não fornecido
    if args.modelo is None:
        now = datetime.now()
        args.modelo = f"SAC_{now.strftime('%Y%m%d')}-{now.strftime('%H%M%S')}-{args.heuristica}"
    
    return args

def make_env(seed: int = 0, heuristica: str = "nao_holonomica", log_dir: str = None):
    def _init():
        env = ParkingEnv(heuristica=heuristica)
        if log_dir:
            env = Monitor(env, log_dir)
        env.reset(seed=seed)
        return env
    return _init

def make_vector_env(n_envs: int = 8, use_subproc: bool | None = None, heuristica: str = "nao_holonomica", log_dir: str = None):
    if use_subproc is None:
        use_subproc = platform.system() != "Windows"

    env_fns = [make_env(seed=i, heuristica=heuristica, log_dir=log_dir) for i in range(n_envs)]

    if use_subproc:
        try:
            return SubprocVecEnv(env_fns)
        except Exception as e:
            print(f"SubprocVecEnv failed ({e}), falling back to DummyVecEnv.")

    return DummyVecEnv(env_fns)

def train_sac(model_path: str | None = None, total_timesteps: int = 10000, save_every: int | None = None, save_path: str | None = None, save_name: str | None = None, n_envs: int = 8, heuristica: str = "nao_holonomica", log_dir: str = None):
    env = make_vector_env(n_envs=n_envs, heuristica=heuristica, log_dir=log_dir)

    # Define path for normalization stats based on the model path
    if model_path:
        vec_norm_path = model_path.replace(".zip", "") + "_vecnormalize.pkl"
    else:
        vec_norm_path = None

    if model_path is not None and os.path.exists(model_path) and vec_norm_path and os.path.exists(vec_norm_path):
        print(f"Carregando estatisticas de normalização de {vec_norm_path}")
        env = VecNormalize.load(vec_norm_path, env) # Applied manually AFTER vectorization
        env.training = True      # Continue updating running mean/std
        env.norm_reward = False  # Do not normalize rewards
    else:
        print(f"Criando novas estatisticas de normalização (não encontrado em {vec_norm_path})")
        env = VecNormalize(env, training=True, norm_reward=False)

    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=dict(pi=[256, 128, 128], qf=[256, 128, 128])
    )

    if model_path is None or not os.path.exists(model_path):
        model = SAC(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            tensorboard_log=log_dir,
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
        print(f"Carregando modelo existente de {model_path}")
        model = SAC.load(model_path, env=env)

    timesteps_split = 0
    if save_every is not None:
        timesteps_split = int(total_timesteps / save_every)
        for i in range(timesteps_split):
            print(f"Treinando... {i+1} de {timesteps_split} timesteps")
            model.learn(total_timesteps=save_every, progress_bar=True, tb_log_name=save_name, reset_num_timesteps=False)
            
            if save_path and save_name:
                model_save_path = os.path.join(save_path, save_name)
                # Construct unique path for this model's stats
                vec_save_path = os.path.join(save_path, save_name + "_vecnormalize.pkl")
                
                #salvar modelo e estatisticas de normalização separadamente
                model.save(model_save_path)
                env.save(vec_save_path)
                print(f"modelo salvo em {model_save_path}")
                print(f"Estatisticas de normalização salvas em {vec_save_path}")

    return model

if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    print(f"Começando treinamento com modelo {args.modelo}...")
    
    model_path = os.path.join(args.model_dir, args.modelo + ".zip")

    print(f"""Treinando:
        - modelo: {args.modelo}
        - heurística: {args.heuristica}
        - total de timesteps: {args.timesteps:,}
        - salvar a cada: {args.salvar_cada:,} timesteps
        - número de ambientes: {args.n_envs}
        - diretório de logs: {args.log_dir}
        - diretório de modelos: {args.model_dir}
    """)
    
    train_sac(
        model_path=model_path, 
        total_timesteps=args.timesteps, 
        save_every=args.salvar_cada, 
        save_path=args.model_dir, 
        save_name=args.modelo,
        n_envs=args.n_envs,
        heuristica=args.heuristica,
        log_dir=args.log_dir
    )
