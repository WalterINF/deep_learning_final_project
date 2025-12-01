# dominio/visualizacoes/v_mapa.py
import os
import numpy as np
import matplotlib.pyplot as plt
from interfaces.i_visualizacao import IVisualizarMapa


class VMapa(IVisualizarMapa):
    """
    Visualizador de mapas com coordenadas métricas globais.
    Mostra sempre o mapa completo (sem cortes) e permite sobrepor caminhos.
    """

    def __init__(self):
        pass

    # ============================================================
    # Conversões de coordenadas
    # ============================================================
    @staticmethod
    def meters_to_pixels(x_m, y_m, zero_x, zero_y, resolution=16):
        """Converte coordenadas em metros para pixels (imagem cortada)."""
        pixel_x = zero_x + (x_m * resolution)
        pixel_y = zero_y - (y_m * resolution)  # y invertido
        return pixel_x, pixel_y

    @staticmethod
    def pixels_to_meters(pixel_x, pixel_y, zero_x, zero_y, resolution=16):
        """Converte coordenadas de pixels para metros no sistema global."""
        x_m = (pixel_x - zero_x) / resolution
        y_m = (zero_y - pixel_y) / resolution
        return x_m, y_m

    # ============================================================
    # Desenhar o mapa base
    # ============================================================
    def desenhar_mapa(self, ax, mapa):
        """
        Plota o mapa completo em coordenadas métricas,
        com eixos em metros e origem marcada.
        """
        if hasattr(mapa, "obterMapa"):
            mapa_data = mapa.obterMapa()
            zero_x, zero_y = mapa.ponto_de_referencia_global
            resolucao = mapa.resolucao
            nome = mapa.nome
        else:
            raise ValueError("O objeto de mapa deve conter obterMapa(), ponto_de_referencia_global e resolucao.")

        height, width = mapa_data.shape

        # Limites em metros correspondentes ao mapa inteiro
        x_min_m, y_max_m = self.pixels_to_meters(0, 0, zero_x, zero_y, resolucao)
        x_max_m, y_min_m = self.pixels_to_meters(width - 1, height - 1, zero_x, zero_y, resolucao)

        # Exibir mapa completo (imagem inteira)
        ax.imshow(mapa_data, cmap="gray", origin="upper")

        # Ticks e labels em metros
        passo_m = 10  # espaçamento em metros entre marcações
        x_ticks_pixels = np.arange(0, width, resolucao * passo_m)
        y_ticks_pixels = np.arange(0, height, resolucao * passo_m)
        x_ticks_meters = [(x - zero_x) / resolucao for x in x_ticks_pixels]
        y_ticks_meters = [(zero_y - y) / resolucao for y in y_ticks_pixels]

        ax.set_xticks(x_ticks_pixels)
        ax.set_xticklabels([f"{x:.0f}m" for x in x_ticks_meters])
        ax.set_yticks(y_ticks_pixels)
        ax.set_yticklabels([f"{y:.0f}m" for y in y_ticks_meters])

        # Marcar origem (0,0)
        ax.plot(zero_x, zero_y, "ro", markersize=3, label="Origem (0,0)")

        # Limites sempre abrangendo o mapa inteiro
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.set_aspect("equal", adjustable="box")

        # Grade e rótulos
        ax.grid(True, alpha=0.3)
        ax.set_title(f"Mapa {nome} — Coordenadas Globais (m)")
        ax.set_xlabel("Coordenada X (m)")
        ax.set_ylabel("Coordenada Y (m)")
        ax.legend()

        # Log informativo
        print(f"=== Limites do mapa '{nome}' ===")
        print(f"  X: {x_min_m:.2f} m  →  {x_max_m:.2f} m")
        print(f"  Y: {y_min_m:.2f} m  →  {y_max_m:.2f} m")
        print(f"  Tamanho da imagem: {width} px × {height} px")

        return ax

    # ============================================================
    # Desenhar caminho (em metros)
    # ============================================================
    def desenhar_caminho(self, ax, caminho_metros, mapa):
        """
        Desenha um caminho sobre o mapa completo (em coordenadas métricas).
        """
        if not caminho_metros:
            return ax

        zero_x, zero_y = mapa.ponto_de_referencia_global
        resolucao = mapa.resolucao

        # Converte caminho (m) → (px)
        caminho_px = [self.meters_to_pixels(x, y, zero_x, zero_y, resolucao) for x, y in caminho_metros]
        xs, ys = zip(*caminho_px)

        ax.plot(xs, ys, color='lime', linewidth=1.8, label='Trajetória planejada')
        ax.legend()
        return ax

    # ============================================================
    # Visualização completa
    # ============================================================
    def visualizar(self, mapa, caminho_metros=None):
        """
        Exibe o mapa completo e, opcionalmente, o caminho planejado.
        Sempre mostra o mapa inteiro (sem zoom automático).
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        self.desenhar_mapa(ax, mapa)
        if caminho_metros:
            self.desenhar_caminho(ax, caminho_metros, mapa)

        plt.tight_layout()
        plt.show()
