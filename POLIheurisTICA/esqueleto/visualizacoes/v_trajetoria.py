import os
import matplotlib
from typing import List
matplotlib.use('TkAgg')  # Explicitly set the backend
from collections import deque
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.collections import PatchCollection
from dominio.parametros_trator_trailer import ParametrosTratorTrailer, ParametrosTrator
from interfaces.i_visualizacao import IVisualizaTrajetoria
from dominio.visualizacoes.v_estado import VEstado, VEstadoTrator
from dominio.entidades.estados import Estado7, Estado3

# Defina o caminho para o executável do FFmpeg
# animation.writers['ffmpeg']._executable = "C://Users//00050786//AppData//Local//ffmpeg//bin//ffmpeg.exe"
class VisualizaTrajetoria(IVisualizaTrajetoria):
    def __init__(self):
        self.visualizar_estado = VEstado()

    def desenhar_trajetoria(self, ax, parametros_veiculo: ParametrosTratorTrailer, trajectory: List[Estado7]):
        # Inicializar o visualizador
        # self.visualizar_estado = VEstado()
        patches = []

        for estado in trajectory:
            vertices = parametros_veiculo.calcula_vertices_em_metros(estado)
            trator_patch, trailer_patch = self.visualizar_estado.create_patches(vertices)
            patches.extend([trator_patch, trailer_patch])

        # Criar uma coleção de patches com transparência
        p = PatchCollection(patches, alpha=0.4)
        ax.add_collection(p)

        # plt.show()

        return patches
    
    def desenhar_trajetoria_com_rodas(self, ax, parametros_veiculo: ParametrosTratorTrailer, conjunto : deque):
        # Inicializar o visualizador
        # self.visualizar_estado = VEstado()

        for estado, angle in conjunto:

            vertices = parametros_veiculo.calcular_vertices_com_rodas(estado, angle)
            self.visualizar_estado.desenhar_pose_com_rodas(ax, vertices)

            # plt.show()
            
    def animar_trajetoria(self, ax, parametros_veiculo: ParametrosTratorTrailer, conjunto: deque, nome_arquivo: str, folder: str, modelo):
        # self.visualizar_estado = VEstado()
        caminho_x = []
        caminho_y = []

        # Configurar figura e eixos
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Animação da Trajetória')
        ax.figure.set_size_inches(10, 6)  # Definir tamanho fixo para reduzir carga

        # Pré-calcular todos os vértices para evitar cálculos repetidos
        vertices_pre_calculados = [
            parametros_veiculo.calcular_vertices_com_rodas(estado, angle)
            for estado, angle, _velocidade in conjunto
        ]

        # Inicializar elementos gráficos
        caminho_linha, = ax.plot([], [], 'b-', label='Caminho')
        veiculo_patch = self.visualizar_estado.desenhar_pose_com_rodas(ax, vertices_pre_calculados[0])

        def atualizar(frame):
            nonlocal veiculo_patch

            # Remover patches antigos
            for patch in veiculo_patch:
                patch.remove()

            # Usar vértices pré-calculados
            vertices = vertices_pre_calculados[frame]
            veiculo_patch = self.visualizar_estado.desenhar_pose_com_rodas(ax, vertices)

            # Atualizar caminho
            estado, _, _velocidade = conjunto[frame] # Obter o estado completo
            caminho_x.append(estado.obter_x())
            caminho_y.append(estado.obter_y())
            caminho_linha.set_data(caminho_x, caminho_y)

            return list(veiculo_patch) + [caminho_linha]

        # Configurar o escritor FFmpeg com parâmetros otimizados
        writer = FFMpegWriter(fps=60, metadata={'artist': 'Trajetória'}, bitrate=1000)

        # Salvar a animação
        print('Salvando a animação...')
        output_path = os.path.join(folder, f"{nome_arquivo}.mp4")
        
        # Reduzir número de frames para teste (opcional, comente se não desejar)
        frames = min(len(conjunto), 3000)  # Limita a 300 frames como exemplo
        
        ani = FuncAnimation(ax.figure, atualizar, frames=frames, interval=66, blit=True, repeat=False)

        # return ax, ani
        ani.save(output_path, writer=writer)
        # # plt.legend()
        
        print(f'Animação salva em: {output_path}')
        # plt.show()
        # os.startfile(output_path)  # Abre o vídeo após salvar

        #################################################
        
class VisualizaTrajetoriaTrator(IVisualizaTrajetoria):
    def __init__(self):
        self.visualizar_estado = VEstadoTrator()

    def desenhar_trajetoria(self, ax, parametros_veiculo: ParametrosTratorTrailer, trajectory: List[Estado3]):
        # Inicializar o visualizador
        # self.visualizar_estado = VEstado()
        patches = []

        for estado in trajectory:
            vertices = parametros_veiculo.calcula_vertices_em_metros(estado)
            trator_patch = self.visualizar_estado.create_patches(vertices)
            patches.append(trator_patch)

        # Criar uma coleção de patches com transparência
        p = PatchCollection(patches, alpha=0.4)
        ax.add_collection(p)

        # plt.show()

        return patches
    
    def desenhar_trajetoria_com_rodas(self, ax, parametros_veiculo: ParametrosTrator, conjunto : deque):
        # Inicializar o visualizador
        # self.visualizar_estado = VEstado()

        for estado, angle in conjunto:

            vertices = parametros_veiculo.calcular_vertices_com_rodas(estado, angle)
            self.visualizar_estado.desenhar_pose_com_rodas(ax, vertices)

            # plt.show()
            
    def animar_trajetoria(self, ax, parametros_veiculo: ParametrosTrator, conjunto: deque, nome_arquivo: str, folder: str, modelo=None):
        # self.visualizar_estado = VEstado()
        caminho_x = []
        caminho_y = []

        # Configurar figura e eixos
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title('Animação da Trajetória')
        ax.figure.set_size_inches(10, 6)  # Definir tamanho fixo para reduzir carga

        # Pré-calcular todos os vértices para evitar cálculos repetidos
        vertices_pre_calculados = [
            parametros_veiculo.calcular_vertices_com_rodas(estado, angle)
            for estado, angle, _velocidade in conjunto
        ]

        # Inicializar elementos gráficos
        caminho_linha, = ax.plot([], [], 'b-', label='Caminho')
        veiculo_patch = self.visualizar_estado.desenhar_pose_com_rodas(ax, vertices_pre_calculados[0])

        def atualizar(frame):
            nonlocal veiculo_patch

            # Remover patches antigos
            for patch in veiculo_patch:
                patch.remove()

            # Usar vértices pré-calculados
            vertices = vertices_pre_calculados[frame]
            veiculo_patch = self.visualizar_estado.desenhar_pose_com_rodas(ax, vertices)

            # Atualizar caminho
            estado, _, _velocidade = conjunto[frame]
            caminho_x.append(estado.obter_x())
            caminho_y.append(estado.obter_y())
            caminho_linha.set_data(caminho_x, caminho_y)

            return list(veiculo_patch) + [caminho_linha]

        # Configurar o escritor FFmpeg com parâmetros otimizados
        writer = FFMpegWriter(fps=60, metadata={'artist': 'Trajetória'}, bitrate=1000)

        # Salvar a animação
        print('Salvando a animação...')
        output_path = os.path.join(folder, f"{nome_arquivo}.mp4")
        
        # Reduzir número de frames para teste (opcional, comente se não desejar)
        frames = min(len(conjunto), 3000)  # Limita a 300 frames como exemplo
        
        ani = FuncAnimation(ax.figure, atualizar, frames=frames, interval=66, blit=True, repeat=False)

        # return ax, ani
        ani.save(output_path, writer=writer)
        # # plt.legend()
        
        print(f'Animação salva em: {output_path}')
        # plt.show()
        # os.startfile(output_path)  # Abre o vídeo após salvar

        