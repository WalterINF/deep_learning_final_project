#!/usr/bin/env python3
"""
Script to load TensorBoard logs and plot episode reward and success rate.
Uses tensorboard's EventAccumulator to read event files.
"""

import os
import sys
import re
import pathlib
import argparse
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Configuration
SCRIPT_DIR = pathlib.Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "plots"

# Default experiments to plot (if None, plot all)
DEFAULT_EXPERIMENTS = None

# X-axis cutoff (in steps). Set to None for no cutoff.
X_AXIS_CUTOFF = 3_500_000


def generate_alias(experiment_name: str) -> str:
    """Generate a display alias by removing all numbers and hyphens from the experiment name."""
    # Remove all digits and hyphens
    alias = re.sub(r'[\d\-]', '', experiment_name)
    # Clean up multiple consecutive underscores
    alias = re.sub(r'_+', '_', alias)
    # Remove leading/trailing underscores
    alias = alias.strip('_')
    return alias

# Auto-save plots
AUTO_SAVE_PLOTS = True


def get_experiment_dirs(log_dir: pathlib.Path) -> list[pathlib.Path]:
    """Get all experiment directories containing TensorBoard event files."""
    experiment_dirs = []
    for item in log_dir.iterdir():
        if item.is_dir():
            # Check if directory contains event files
            event_files = list(item.glob("events.out.tfevents.*"))
            if event_files:
                experiment_dirs.append(item)
    return sorted(experiment_dirs, key=lambda x: x.name)


def load_tensorboard_logs(experiment_dir: pathlib.Path) -> dict:
    """Load all scalar metrics from a TensorBoard experiment directory."""
    event_acc = EventAccumulator(str(experiment_dir))
    event_acc.Reload()
    
    metrics = {}
    scalar_tags = event_acc.Tags().get('scalars', [])
    
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        metrics[tag] = {'steps': np.array(steps), 'values': np.array(values)}
    
    return metrics


def smooth_data(values: np.ndarray, window: int = 10) -> np.ndarray:
    """Apply moving average smoothing to data."""
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode='valid')
    # Pad the beginning to maintain array length
    pad_size = len(values) - len(smoothed)
    return np.concatenate([values[:pad_size], smoothed])


def plot_metric(ax, experiments_data: dict, metric_name: str, title: str, 
                ylabel: str, smooth_window: int = 10, show_raw: bool = True,
                x_cutoff: int = None):
    """Plot a single metric for multiple experiments."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments_data)))
    
    for (exp_name, metrics), color in zip(experiments_data.items(), colors):
        if metric_name not in metrics:
            continue
            
        steps = metrics[metric_name]['steps']
        values = metrics[metric_name]['values']
        
        if len(steps) == 0:
            continue
        
        # Apply x-axis cutoff if specified
        if x_cutoff is not None:
            mask = steps <= x_cutoff
            steps = steps[mask]
            values = values[mask]
        
        if len(steps) == 0:
            continue
        
        # Plot raw data with transparency
        if show_raw:
            ax.plot(steps, values, alpha=0.2, color=color, linewidth=0.5)
        
        # Plot smoothed data
        smoothed = smooth_data(values, smooth_window)
        ax.plot(steps, smoothed, label=exp_name, color=color, linewidth=1.5)
    
    ax.set_xlabel('Timesteps')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Set x-axis limit
    if x_cutoff is not None:
        ax.set_xlim(0, x_cutoff)


def plot_all_metrics(experiments_data: dict, output_dir: pathlib.Path = None,
                     smooth_window: int = 10, show_raw: bool = True):
    """Create a comprehensive plot with episode reward and success rate."""
    
    # Collect all available metrics across experiments
    all_metrics = set()
    for metrics in experiments_data.values():
        all_metrics.update(metrics.keys())
    
    # Define metrics to plot with their save names
    metrics_to_plot = []
    
    # Episode reward - check various possible names
    reward_metrics = ['rollout/ep_rew_mean', 'train/reward', 'episode_reward']
    for m in reward_metrics:
        if m in all_metrics:
            metrics_to_plot.append((m, 'Episode Reward', 'Mean Reward', 'episode_reward'))
            break
    
    # Success rate - check various possible names
    success_metrics = ['rollout/success_rate', 'success_rate', 'train/success_rate', 
                       'eval/success_rate', 'is_success']
    for m in success_metrics:
        if m in all_metrics:
            metrics_to_plot.append((m, 'Success Rate', 'Success Rate', 'success_rate'))
            break
    
    # Episode length
    length_metrics = ['rollout/ep_len_mean', 'episode_length']
    for m in length_metrics:
        if m in all_metrics:
            metrics_to_plot.append((m, 'Episode Length', 'Mean Episode Length', 'episode_length'))
            break
    
    # Actor loss
    actor_loss_metrics = ['train/actor_loss', 'actor_loss', 'train/policy_loss', 'policy_loss']
    for m in actor_loss_metrics:
        if m in all_metrics:
            metrics_to_plot.append((m, 'Actor Loss', 'Loss', 'actor_loss'))
            break
    
    # Critic loss
    critic_loss_metrics = ['train/critic_loss', 'critic_loss', 'train/value_loss', 'value_loss', 'train/qf_loss']
    for m in critic_loss_metrics:
        if m in all_metrics:
            metrics_to_plot.append((m, 'Critic Loss', 'Loss', 'critic_loss'))
            break
    
    if not metrics_to_plot:
        print("No plottable metrics found!")
        print(f"Available metrics: {sorted(all_metrics)}")
        return
    
    # Get experiment names for filename
    exp_names = "_vs_".join(experiments_data.keys())
    
    # Create output directory if saving
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot each metric separately for better quality
    for metric_name, title, ylabel, save_name in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_metric(ax, experiments_data, metric_name, title, ylabel, 
                   smooth_window, show_raw, x_cutoff=X_AXIS_CUTOFF)
        
        plt.tight_layout()
        
        if output_dir:
            filename = f"{save_name}_{exp_names}.png"
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.show()
    
    # Also create combined plot
    n_plots = len(metrics_to_plot)
    if n_plots > 1:
        fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
        
        for ax, (metric_name, title, ylabel, _) in zip(axes, metrics_to_plot):
            plot_metric(ax, experiments_data, metric_name, title, ylabel, 
                       smooth_window, show_raw, x_cutoff=X_AXIS_CUTOFF)
        
        plt.suptitle('Training Progress Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_dir:
            filename = f"training_comparison_{exp_names}.png"
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.show()


def list_available_metrics(experiments_data: dict):
    """List all available metrics from loaded experiments."""
    all_metrics = defaultdict(int)
    for exp_name, metrics in experiments_data.items():
        for metric_name in metrics.keys():
            all_metrics[metric_name] += 1
    
    print("\nAvailable metrics (count = number of experiments with this metric):")
    print("-" * 60)
    for metric, count in sorted(all_metrics.items()):
        print(f"  {metric}: {count} experiments")


def main():
    parser = argparse.ArgumentParser(
        description='Plot TensorBoard logs for episode reward and success rate'
    )
    parser.add_argument(
        '--log-dir', '-l', type=str, default=str(LOG_DIR),
        help=f'Directory containing TensorBoard logs (default: {LOG_DIR})'
    )
    parser.add_argument(
        '--output', '-o', type=str, default=None,
        help=f'Output directory for plots (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--experiments', '-e', nargs='+', type=str, default=None,
        help='Specific experiments to plot (default: all)'
    )
    parser.add_argument(
        '--smooth', '-s', type=int, default=10,
        help='Smoothing window size (default: 10)'
    )
    parser.add_argument(
        '--no-raw', action='store_true',
        help='Hide raw (unsmoothed) data'
    )
    parser.add_argument(
        '--list-metrics', action='store_true',
        help='List available metrics and exit'
    )
    parser.add_argument(
        '--metric', '-m', type=str, default=None,
        help='Plot a specific metric by name'
    )
    
    args = parser.parse_args()
    
    log_dir = pathlib.Path(args.log_dir)
    if not log_dir.exists():
        print(f"Error: Log directory does not exist: {log_dir}")
        sys.exit(1)
    
    # Find experiment directories
    experiment_dirs = get_experiment_dirs(log_dir)
    
    if not experiment_dirs:
        print(f"No TensorBoard experiments found in: {log_dir}")
        sys.exit(1)
    
    print(f"Found {len(experiment_dirs)} experiments:")
    for exp_dir in experiment_dirs:
        print(f"  - {exp_dir.name}")
    
    # Filter experiments if specified, otherwise use defaults
    experiments_filter = args.experiments if args.experiments else DEFAULT_EXPERIMENTS
    if experiments_filter:
        experiment_dirs = [
            d for d in experiment_dirs 
            if any(exp in d.name for exp in experiments_filter)
        ]
        print(f"\nFiltered to {len(experiment_dirs)} experiments")
    
    # Load data from all experiments
    print("\nLoading TensorBoard logs...")
    experiments_data = {}
    for exp_dir in experiment_dirs:
        try:
            metrics = load_tensorboard_logs(exp_dir)
            if metrics:
                # Auto-generate display name by removing numbers and hyphens
                display_name = generate_alias(exp_dir.name)
                experiments_data[display_name] = metrics
                print(f"  Loaded {exp_dir.name} as '{display_name}': {len(metrics)} metrics")
        except Exception as e:
            print(f"  Warning: Failed to load {exp_dir.name}: {e}")
    
    if not experiments_data:
        print("No data could be loaded!")
        sys.exit(1)
    
    # List metrics if requested
    if args.list_metrics:
        list_available_metrics(experiments_data)
        sys.exit(0)
    
    # Determine output directory
    if args.output:
        output_dir = pathlib.Path(args.output)
    elif AUTO_SAVE_PLOTS:
        output_dir = OUTPUT_DIR
    else:
        output_dir = None
    
    if args.metric:
        # Plot single specific metric
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_metric(ax, experiments_data, args.metric, args.metric, 
                   'Value', args.smooth, not args.no_raw, x_cutoff=X_AXIS_CUTOFF)
        plt.tight_layout()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            exp_names = "_vs_".join(experiments_data.keys())
            metric_clean = args.metric.replace('/', '_')
            filename = f"{metric_clean}_{exp_names}.png"
            output_path = output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {output_path}")
        
        plt.show()
    else:
        # Plot default metrics (reward, success rate, episode length)
        plot_all_metrics(
            experiments_data, 
            output_dir,
            smooth_window=args.smooth,
            show_raw=not args.no_raw
        )


if __name__ == "__main__":
    main()

