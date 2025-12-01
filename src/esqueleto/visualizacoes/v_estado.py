from matplotlib.patches import Polygon
from interfaces.i_visualizacao import IVisualizaEstado
import matplotlib.pyplot as plt
import numpy as np
cos = np.cos
sin = np.sin

class VEstado(IVisualizaEstado):
    def __init__(self):
        pass

    def desenhar_pose(self, ax, vertices: dict, colors=['darkturquoise', 'steelblue']):
        """
        Plota o veículo como um polígono baseado nos vértices fornecidos.

        Parâmetros:
        ----------
        ax (object):
            Eixo do matplotlib onde o veículo será plotado.
        vertices (dict): 
            Dicionário com as coordenadas dos vértices do trator e do trailer 
        cor (str):
            Cor do polígono que representa o veículo.
        """

        # Desenhar o trator
        trator = plt.Polygon(vertices['trator'], closed=True, color=colors[0])
        ax.add_patch(trator)
        
        # Desenhar o trailer
        trailer = plt.Polygon(vertices['trailer'], closed=True, color=colors[1])
        ax.add_patch(trailer)

        return trator, trailer
    
    def create_patches(self, vertices):
        tractor_vertices = vertices['trator']
        trailer_vertices = vertices['trailer']

        trator_patch = Polygon(tractor_vertices, closed=True, color='darkturquoise')
        trailer_patch = Polygon(trailer_vertices, closed=True, color='steelblue')

        return trator_patch, trailer_patch
    
    def desenhar2poses(self, ax, vertices1: dict, vertices2: dict, colors=['darkturquoise', 'steelblue', 'gold']):
        """
        Plota dois veículos como polígonos baseados nos vértices fornecidos.

        Parâmetros:
        ----------
        ax : object
            Eixo do matplotlib onde o veículo será plotado.
        vertices1 : list of tuples
            Lista de coordenadas (x, y) dos vértices do primeiro veículo.
        vertices2 : list of tuples
            Lista de coordenadas (x, y) dos vértices do segundo veículo.
        cor : str
            Cor do polígono que representa o veículo.
        """

        # Desenhar veiculo 1
        trator1, trailer1 = self.desenhar_pose(ax, vertices1, [colors[0], colors[1]])

        # Desenhar veiculo 2
        trator2, trailer2 = self.desenhar_pose(ax, vertices2, [colors[-1], colors[0]])

        return trator1, trailer1, trator2, trailer2
    

    def desenhar_pose_com_rodas(self, ax, vertices: dict, colors=['darkturquoise', 'steelblue', 'gold']):
        """
        Plota o veículo como um polígono baseado nos vértices fornecidos.

        Parâmetros:
        ----------
        ax (object):
            Eixo do matplotlib onde o veículo será plotado.
        vertices (dict): 
            Dicionário com as coordenadas dos vértices do trator, do trailer e das rodas.
        cor (str):
            Cor do polígono que representa o veículo.
        """

        # Desenhar o veiculo
        trator, trailer = self.desenhar_pose(ax, vertices, [colors[0], colors[1]])

        # Desenhar as rodas
        roda_dianteira_esquerda = plt.Polygon(vertices['roda_dianteira_esquerda'], closed=True, color=colors[2])
        ax.add_patch(roda_dianteira_esquerda)
        
        roda_dianteira_direita = plt.Polygon(vertices['roda_dianteira_direita'], closed=True, color=colors[2])
        ax.add_patch(roda_dianteira_direita)

        return trator, trailer, roda_dianteira_esquerda, roda_dianteira_direita

class VEstadoTrator(IVisualizaEstado):
    def __init__(self):
        pass

    def desenhar_pose(self, ax, vertices: dict, colors=['darkturquoise', 'steelblue']):
        """
        Plota o veículo como um polígono baseado nos vértices fornecidos.

        Parâmetros:
        ----------
        ax (object):
            Eixo do matplotlib onde o veículo será plotado.
        vertices (dict): 
            Dicionário com as coordenadas dos vértices do trator e do trailer 
        cor (str):
            Cor do polígono que representa o veículo.
        """

        # Desenhar o trator
        trator = plt.Polygon(vertices['trator'], closed=True, color=colors[0])
        ax.add_patch(trator)

        return trator
    
    def create_patches(self, vertices):
        tractor_vertices = vertices['trator']

        trator_patch = Polygon(tractor_vertices, closed=True, color='darkturquoise')

        return trator_patch
    
    def desenhar2poses(self, ax, vertices1: dict, vertices2: dict, colors=['darkturquoise', 'steelblue', 'gold']):
        """
        Plota dois veículos como polígonos baseados nos vértices fornecidos.

        Parâmetros:
        ----------
        ax : object
            Eixo do matplotlib onde o veículo será plotado.
        vertices1 : list of tuples
            Lista de coordenadas (x, y) dos vértices do primeiro veículo.
        vertices2 : list of tuples
            Lista de coordenadas (x, y) dos vértices do segundo veículo.
        cor : str
            Cor do polígono que representa o veículo.
        """

        # Desenhar veiculo 1
        trator1 = self.desenhar_pose(ax, vertices1, [colors[0], colors[1]])

        # Desenhar veiculo 2
        trator2 = self.desenhar_pose(ax, vertices2, [colors[-1], colors[0]])

        return trator1, trator2


    def desenhar_pose_com_rodas(self, ax, vertices: dict, colors=['darkturquoise', 'steelblue', 'gold']):
        """
        Plota o veículo como um polígono baseado nos vértices fornecidos.

        Parâmetros:
        ----------
        ax (object):
            Eixo do matplotlib onde o veículo será plotado.
        vertices (dict): 
            Dicionário com as coordenadas dos vértices do trator, do trailer e das rodas.
        cor (str):
            Cor do polígono que representa o veículo.
        """

        # Desenhar o veiculo
        trator = self.desenhar_pose(ax, vertices, [colors[0], colors[1]])

        # Desenhar as rodas
        roda_dianteira_esquerda = plt.Polygon(vertices['roda_dianteira_esquerda'], closed=True, color=colors[2])
        ax.add_patch(roda_dianteira_esquerda)
        
        roda_dianteira_direita = plt.Polygon(vertices['roda_dianteira_direita'], closed=True, color=colors[2])
        ax.add_patch(roda_dianteira_direita)

        return trator, roda_dianteira_esquerda, roda_dianteira_direita
    