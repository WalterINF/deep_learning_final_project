import matplotlib.pyplot as plt
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from src.interfaces.i_visualizacao import IVisualizaCaminho

class VCaminho(IVisualizaCaminho):
    def __init__(self):
        pass
    def desenhar_caminho(self, ax, path):
        # plotar o path no plot principal
        # line dashdot style
        ax.plot([p[0] for p in path], [p[1] for p in path], linewidth=2)
        # plt.show()

        
if __name__ == "__main__":

    fig, ax = plt.subplots()
    path = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    VCaminho.desenhar_caminho(ax, path)
