from interfaces.i_visualizacao import ISeta
import numpy as np

class VSeta(ISeta):
    def __init__(self):
        pass

    def desenhar_seta(self, ax, start, length=2, width=2):
        """ Function to draw an arrow representing the state on the figure."""
        start = start.obter_estado_em_metros()
        # Draw the arrow
        ax.arrow(
            start[0], start[1],
            length * np.cos(start[2]),
            length * np.sin(start[2]),
            head_width=width, head_length=width, fc='k', ec='k'
        )
        return ax

