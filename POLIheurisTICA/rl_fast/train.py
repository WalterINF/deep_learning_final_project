from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

from .fast_env import FastParkingEnv


def make_env(seed: int):
    def _init() -> FastParkingEnv:
        env = FastParkingEnv(seed=seed)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAC agent for articulated parking.")
    parser.add_argument("--num-envs", type=int, default=10, help="Número de ambientes paralelos.")
    parser.add_argument("--total-steps", type=int, default=5_000_000, help="Total de timesteps para treinamento.")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("rl_fast/logs"),
        help="Diretório de logs/TensorBoard.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Seleção manual do dispositivo PyTorch.",
    )
    return parser.parse_args()


def record_evaluation_video(model: SAC, output_dir: Path, seed: int = 0, fps: int = 10, max_steps: int = 800) -> None:
    try:
        import imageio.v2 as imageio
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError as exc:
        print(f"[video] Dependência ausente, pulando geração de vídeo: {exc}")
        return

    env = FastParkingEnv(seed=seed)
    observation, info = env.reset()
    sim_result = info.get("sim_result")
    if sim_result is None:
        print("[video] Não foi possível obter o estado inicial; vídeo não será gerado.")
        env.close()
        return

    base_map = env.sim._map_matrix  # type: ignore[attr-defined]
    rendered_map = np.where(base_map > 0, 235, 40).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    frames: List[np.ndarray] = []
    trajectory: List[np.ndarray] = [sim_result.state[:2].copy()]

    window_meters = 30.0
    half_window_px = int(env.sim.px_per_meter * (window_meters / 2.0))

    def draw_frame(current_result) -> np.ndarray:
        ax.clear()
        ax.imshow(rendered_map, cmap="gray", origin="upper")

        if trajectory:
            pixels = [
                env.sim.map.coordenadaGlobalParaPixel((float(pos[0]), float(pos[1])))
                for pos in trajectory
            ]
            xs, ys = zip(*pixels)
            ax.plot(xs, ys, color="#00ffff", linewidth=1.2, alpha=0.7)

        for name, polygon in current_result.polygons.items():
            pixel_poly = np.array(
                [
                    env.sim.map.coordenadaGlobalParaPixel((float(pt[0]), float(pt[1])))
                    for pt in polygon
                ],
                dtype=float,
            )
            color = "#ffb347" if name.lower().startswith("trator") else "#3fa7dc"
            ax.add_patch(
                Polygon(
                    pixel_poly,
                    closed=True,
                    facecolor=color,
                    edgecolor="#1b1b1b",
                    linewidth=1.0,
                    alpha=0.75,
                )
            )

        rear_pose_px = env.sim.map.coordenadaGlobalParaPixel(
            (float(current_result.state[0]), float(current_result.state[1]))
        )
        ax.scatter(rear_pose_px[0], rear_pose_px[1], color="#ff0054", s=18, zorder=6)

        heading = float(current_result.state[2])
        arrow_length_px = env.sim.px_per_meter * 3.0
        ax.annotate(
            "",
            xy=(rear_pose_px[0] + math.cos(heading) * arrow_length_px,
                rear_pose_px[1] - math.sin(heading) * arrow_length_px),
            xytext=rear_pose_px,
            arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#ff0054"),
        )

        if env.sim.goal is not None:
            goal_px = env.sim.map.coordenadaGlobalParaPixel(
                (float(env.sim.goal.center[0]), float(env.sim.goal.center[1]))
            )
            ax.scatter(
                goal_px[0],
                goal_px[1],
                color="#ffd700",
                marker="*",
                s=90,
                edgecolors="#000000",
                linewidths=0.8,
                zorder=5,
            )

        cx, cy = rear_pose_px
        width, height = rendered_map.shape[1], rendered_map.shape[0]
        xmin = max(0, cx - half_window_px)
        xmax = min(width, cx + half_window_px)
        ymin = max(0, cy - half_window_px)
        ymax = min(height, cy + half_window_px)
        if xmax - xmin < 10:
            xmin = max(0, cx - 10)
            xmax = min(width, cx + 10)
        if ymax - ymin < 10:
            ymin = max(0, cy - 10)
            ymax = min(height, cy + 10)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Fast Parking Evaluation", fontsize=10)

        canvas = fig.canvas
        canvas.draw()
        buffer = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore[attr-defined]
        image = buffer.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        return image

    frames.append(draw_frame(sim_result))

    terminated = truncated = False
    step_counter = 0
    while not (terminated or truncated) and step_counter < max_steps:
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        sim_result = info.get("sim_result")
        if sim_result is None:
            break
        trajectory.append(sim_result.state[:2].copy())
        frames.append(draw_frame(sim_result))
        step_counter += 1

    env.close()
    plt.close(fig)

    if not frames:
        print("[video] Nenhum quadro capturado; vídeo não será criado.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    video_path = output_dir / f"sac_fast_parking_{timestamp}.mp4"

    try:
        with imageio.get_writer(video_path, fps=fps) as writer:
            for frame in frames:
                writer.append_data(frame)  # type: ignore[attr-defined]
        print(f"[video] Vídeo de avaliação salvo em {video_path}")
    except Exception as exc:
        gif_path = video_path.with_suffix(".gif")
        try:
            imageio.mimsave(gif_path, frames, duration=1.0 / fps)  # type: ignore[arg-type]
            print(f"[video] MP4 falhou ({exc}); GIF salvo em {gif_path}")
        except Exception as gif_exc:
            print(f"[video] Falha ao salvar vídeo: {exc}; fallback GIF também falhou: {gif_exc}")


def main() -> None:
    args = parse_args()
    args.log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    vec_env = SubprocVecEnv([make_env(seed=i) for i in range(args.num_envs)])

    train_freq = (1, "episode") if args.num_envs == 1 else (1, "step")

    model = SAC(
        "MlpPolicy",
        vec_env,
        device=device,
        batch_size=2048,
        buffer_size=1_000_000,
        learning_starts=10_000,
        train_freq=train_freq,
        gradient_steps=-1,
        ent_coef="auto",
        verbose=1,
        tensorboard_log=str(args.log_dir),
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=str(checkpoint_dir),
        name_prefix="sac_fast_parking",
        save_replay_buffer=True,
    )

    model.learn(total_timesteps=args.total_steps, callback=checkpoint_callback)
    model.save(str(args.log_dir / "sac_fast_parking_final"))
    vec_env.close()
    record_evaluation_video(model, args.log_dir / "videos")


if __name__ == "__main__":
    main()
