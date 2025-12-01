import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.SimulationConfigLoader import SimulationLoader
from src.Visualization import to_rgb_array
from src.ParkingEnv import ParkingEnv
import matplotlib.pyplot as plt

def plot_maps():
    simulation_loader = SimulationLoader()
    simulation_1 = simulation_loader.load_simulation()
    rgb_array_1 = to_rgb_array(simulation_1, img_size=(2000, 2000))
    env1 = ParkingEnv()
    rgb_array_2 = to_rgb_array(env1.simulation, img_size=(2000, 2000), distance_map=env1.distance_map)
    
    # Ensure plots directory exists
    plots_dir = os.path.join(project_root, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Save first image
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_array_1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'simulation_map_1.png'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Save second image
    plt.figure(figsize=(20, 20))
    plt.imshow(rgb_array_2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'simulation_map_2.png'), dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    print(f"Images saved to {os.path.join(plots_dir, 'simulation_map_1.png')} and {os.path.join(plots_dir, 'simulation_map_2.png')}")

if __name__ == "__main__":
    plot_maps()
