"""Gera figuras compostas e gráficos para o artigo e os slides a partir dos resultados.

Uso:
    python gerar_figuras_artigo.py --resultados resultados --saida resultados/figuras
"""
import argparse
import json
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

from robinson import CORES_DIRECOES_BGR, NOMES_DIRECOES

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
    'axes.edgecolor': '#CBD5E1',
    'axes.linewidth': 0.8,
})


def ler_rgb(caminho):
    img = cv.imread(str(caminho), cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(caminho)
    if img.ndim == 2:
        return img
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)


def gerar_diagrama_mascaras(saida):
    """Diagrama dos 4 pares anti-simétricos (5 níveis) e equivalência R_{k+4} = -R_k."""
    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=220)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.8)
    ax.axis('off')

    pares = [
        ('R0 (N) / R4 (S)', 'Borda horizontal (—)', [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], 0),
        ('R1 (NO) / R5 (SE)', 'Borda diagonal (/)', [[2, 1, 0], [1, 0, -1], [0, -1, -2]], 1),
        ('R2 (O) / R6 (L)', 'Borda vertical (|)', [[1, 0, -1], [2, 0, -2], [1, 0, -1]], 2),
        ('R3 (SO) / R7 (NE)', r'Borda diagonal (\)', [[0, -1, -2], [1, 0, -1], [2, 1, 0]], 3),
    ]

    for idx, (titulo, sub, mat, cor_idx) in enumerate(pares):
        cx = 1.35 + idx * 2.55
        cor_rgb = CORES_DIRECOES_BGR[cor_idx][::-1] / 255.0
        cor_op_rgb = CORES_DIRECOES_BGR[cor_idx + 4][::-1] / 255.0

        card = FancyBboxPatch(
            (cx - 1.12, 0.35), 2.24, 4.05,
            boxstyle='round,pad=0.08,rounding_size=0.14',
            facecolor='#F8FAFC', edgecolor='#CBD5E1', linewidth=1.1,
        )
        ax.add_patch(card)
        ax.text(cx, 4.15, titulo, ha='center', va='center', fontsize=10.5, fontweight='bold', color='#0F172A')
        ax.text(cx, 3.82, sub, ha='center', va='center', fontsize=8.8, color='#475569')

        # Grade 3x3
        tamanho = 0.56
        x_ini = cx - 1.5 * tamanho
        y_ini = 1.80
        for r in range(3):
            for c in range(3):
                val = mat[r][c]
                bg = '#DBEAFE' if val > 0 else ('#FEE2E2' if val < 0 else '#E2E8F0')
                fg = '#1E3A8A' if val > 0 else ('#991B1B' if val < 0 else '#334155')
                rect = plt.Rectangle(
                    (x_ini + c * tamanho, y_ini + (2 - r) * tamanho),
                    tamanho, tamanho, facecolor=bg, edgecolor='#94A3B8', linewidth=0.9,
                )
                ax.add_patch(rect)
                ax.text(
                    x_ini + (c + 0.5) * tamanho,
                    y_ini + (2 - r + 0.5) * tamanho,
                    f'{val:+d}' if val != 0 else '0',
                    ha='center', va='center', fontsize=10.5, fontweight='bold', color=fg,
                )

        # Barras de cor indicando Rk e -Rk
        ax.add_patch(plt.Rectangle((cx - 0.9, 1.18), 0.22, 0.22, facecolor=cor_rgb, edgecolor='none'))
        ax.text(cx - 0.6, 1.29, f'Rk (+) · {NOMES_DIRECOES[cor_idx]}', ha='left', va='center', fontsize=8.3, color='#1E293B')
        ax.add_patch(plt.Rectangle((cx - 0.9, 0.78), 0.22, 0.22, facecolor=cor_op_rgb, edgecolor='none'))
        ax.text(cx - 0.6, 0.89, f'-Rk (-) · {NOMES_DIRECOES[cor_idx+4]}', ha='left', va='center', fontsize=8.3, color='#1E293B')

    fig.suptitle(
        'As 4 máscaras base de Robinson (5 níveis) e seus opostos anti-simétricos (Rk+4 = -Rk)',
        fontsize=12, fontweight='bold', color='#0F172A', y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(saida / 'fig_mascaras_rosa.png', dpi=220)
    plt.close(fig)


def gerar_respostas_4direcoes(resultados, saida):
    paineis = [
        ('5niveis_resp_R0_recorte.png', 'R0 (N/S) — Realça horizonte e travessas horizontais'),
        ('5niveis_resp_R1_recorte.png', 'R1 (NO/SE) — Realça inclinação (/) do telhado e cerca esq.'),
        ('5niveis_resp_R2_recorte.png', 'R2 (O/L) — Realça mourões verticais e paredes da casa'),
        ('5niveis_resp_R3_recorte.png', r'R3 (SO/NE) — Realça inclinação (\) do telhado e cerca dir.'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), dpi=220)
    for ax, (arq, titulo) in zip(axes.flat, paineis):
        img = ler_rgb(resultados / arq)
        ax.imshow(img, cmap='gray', vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=9.6, fontweight='bold', color='#0F172A', pad=6)
        ax.axis('off')
    fig.suptitle(
        'Respostas direcionais isoladas (5 níveis; cinza médio = 0)',
        fontsize=11.5, fontweight='bold', color='#0F172A', y=0.98,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(saida / 'fig_respostas_4direcoes.png', dpi=220)
    plt.close(fig)


def gerar_comparacao_3_vs_5(resultados, exec_json, saida):
    est = exec_json['estatisticas']
    limiar = exec_json['limiar']
    paineis = [
        ('original_cinza_recorte.png', 'Original\n(componente Y)'),
        ('3niveis_mag_recorte.png', f"Robinson 3 níveis — Magnitude\n(teto 765; máx. {est['3niveis']['mag_norm_max']:.1f})"),
        ('5niveis_mag_recorte.png', f"Robinson 5 níveis — Magnitude\n(teto 1020; máx. {est['5niveis']['mag_norm_max']:.1f})"),
        ('3niveis_bin_recorte.png', f"3 níveis binário\n(τ = {limiar:.0f}; {est['3niveis']['pixels_borda_pct']:.2f}% bordas)"),
        ('5niveis_bin_recorte.png', f"5 níveis binário\n(τ = {limiar:.0f}; {est['5niveis']['pixels_borda_pct']:.2f}% bordas)"),
    ]
    fig = plt.figure(figsize=(11.0, 7.2), dpi=220)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1], hspace=0.28, wspace=0.10)
    posicoes = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]

    # Coloca legenda resumida no slot inferior esquerdo (1, 0)
    ax_info = fig.add_subplot(gs[1, 0])
    ax_info.axis('off')
    texto_info = (
        'Comparação 3 vs. 5 níveis\n\n'
        '• 3 níveis ({-1, 0, +1}):\n'
        '  peso uniforme nos 3 vizinhos\n'
        '  (equivalente bússola de Prewitt).\n\n'
        '• 5 níveis ({-2, -1, 0, +1, +2}):\n'
        '  peso dobrado no eixo central,\n'
        '  reforçando resposta ao contorno\n'
        '  principal e suavizando pontas.'
    )
    ax_info.text(0.05, 0.92, texto_info, va='top', ha='left', fontsize=9.2, color='#1E293B',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#F8FAFC', edgecolor='#CBD5E1'))

    for (r, c), (arq, titulo) in zip(posicoes, paineis):
        ax = fig.add_subplot(gs[r, c])
        ax.imshow(ler_rgb(resultados / arq), cmap='gray', vmin=0, vmax=255)
        ax.set_title(titulo, fontsize=9.0, fontweight='bold', color='#0F172A', pad=5)
        ax.axis('off')

    fig.suptitle('Detecção de bordas: Robinson 3 níveis versus 5 níveis', fontsize=12, fontweight='bold', y=0.98)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.04)
    fig.savefig(saida / 'fig_comparacao_3_vs_5.png', dpi=220)
    plt.close(fig)


def gerar_mapa_direcoes_e_distribuicao(resultados, metricas, saida):
    dist5 = metricas['distribuicao_direcoes_pct']['5niveis']
    dist3 = metricas['distribuicao_direcoes_pct']['3niveis']

    fig = plt.figure(figsize=(11.0, 5.2), dpi=220)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.20)

    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(ler_rgb(resultados / '5niveis_dir_recorte.png'))
    ax_img.set_title('Mapa de orientação em 8 sentidos (5 níveis, τ = 20)', fontsize=10.5, fontweight='bold', pad=6)
    ax_img.axis('off')

    ax_bar = fig.add_subplot(gs[0, 1])
    y_pos = np.arange(8)
    vals5 = [dist5[d] for d in NOMES_DIRECOES]
    vals3 = [dist3[d] for d in NOMES_DIRECOES]
    cores_rgb = [CORES_DIRECOES_BGR[i][::-1] / 255.0 for i in range(8)]

    bars = ax_bar.barh(y_pos - 0.17, vals5, height=0.34, color=cores_rgb, edgecolor='#1E293B', linewidth=0.6, label='5 níveis')
    ax_bar.barh(y_pos + 0.17, vals3, height=0.34, color='#94A3B8', alpha=0.65, edgecolor='#475569', linewidth=0.5, label='3 níveis')
    setas = ('↑', '↖', '←', '↙', '↓', '↘', '→', '↗')
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels([f'R{i} · {d} ({s})' for i, (d, s) in enumerate(zip(NOMES_DIRECOES, setas))], fontsize=9.2, fontweight='bold')
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel('Participação entre pixels de borda (%)', fontsize=9.5)
    ax_bar.set_title('Distribuição direcional na imagem inteira', fontsize=10.5, fontweight='bold', pad=6)
    ax_bar.set_xlim(0, max(vals5 + vals3) * 1.18)
    ax_bar.grid(axis='x', linestyle='--', alpha=0.4)
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    for bar, v in zip(bars, vals5):
        ax_bar.text(v + 0.4, bar.get_y() + bar.get_height() / 2, f'{v:.1f}%', va='center', fontsize=8.2, color='#0F172A')
    ax_bar.legend(loc='lower right', fontsize=8.8)

    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.12)
    fig.savefig(saida / 'fig_mapa_direcoes.png', dpi=220)
    plt.close(fig)


def gerar_grafico_distribuicao_direcoes(metricas, saida):
    """Gráfico isolado em estilo editorial sóbrio/serifa para compor a Figura 3 do artigo."""
    dist5 = metricas['distribuicao_direcoes_pct']['5niveis']
    dist3 = metricas['distribuicao_direcoes_pct']['3niveis']

    with plt.rc_context({
        'font.family': 'serif',
        'font.serif': ['Tinos', 'Liberation Serif', 'DejaVu Serif'],
        'axes.edgecolor': '#000000',
        'axes.linewidth': 0.8,
    }):
        fig, ax = plt.subplots(figsize=(5.4, 4.1), dpi=240)
        y_pos = np.arange(8)
        vals5 = [dist5[d] for d in NOMES_DIRECOES]
        vals3 = [dist3[d] for d in NOMES_DIRECOES]

        bars5 = ax.barh(
            y_pos - 0.17, vals5, height=0.33,
            color='#1E1E1E', edgecolor='#000000', linewidth=0.5, label='5 níveis',
        )
        ax.barh(
            y_pos + 0.17, vals3, height=0.33,
            color='#B8B8B8', edgecolor='#333333', linewidth=0.5, label='3 níveis',
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([f'R{i} ({d})' for i, d in enumerate(NOMES_DIRECOES)], fontsize=9.2, color='#000000')
        ax.invert_yaxis()
        ax.set_xlabel('Participação entre pixels de borda (%)', fontsize=9.2, color='#000000')
        ax.set_xlim(0, max(vals5 + vals3) * 1.22)
        ax.tick_params(colors='#000000', labelsize=8.8)
        ax.grid(axis='x', linestyle=':', color='#999999', alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        for bar, v in zip(bars5, vals5):
            ax.text(v + 0.45, bar.get_y() + bar.get_height() / 2, f'{v:.1f}%',
                    va='center', fontsize=8.3, color='#000000')
        ax.legend(loc='lower right', fontsize=8.8, frameon=True, facecolor='#FFFFFF', edgecolor='#CCCCCC')

        fig.tight_layout(pad=0.6)
        fig.savefig(saida / 'fig_distribuicao_direcoes.png', dpi=240, facecolor='#FFFFFF')
        plt.close(fig)


def gerar_realce_cinza_cor(resultados, exec_json, saida):
    ganho = exec_json['ganho']
    paineis = [
        ('original_cinza_recorte.png', 'Original em cinza (Y)'),
        ('5niveis_realce_cinza_recorte.png', f'Realce Robinson 5 níveis (α = {ganho})'),
        ('original_cor_recorte.png', 'Original colorida (BGR)'),
        ('5niveis_realce_cor_recorte.png', f'Realce em Y preservando Cr/Cb (α = {ganho})'),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), dpi=220)
    for ax, (arq, titulo) in zip(axes.flat, paineis):
        img = ler_rgb(resultados / arq)
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(titulo, fontsize=9.8, fontweight='bold', color='#0F172A', pad=6)
        ax.axis('off')
    fig.suptitle('Realce de bordas por magnitude de Robinson (cinza e cores)', fontsize=11.5, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(saida / 'fig_realce_cinza_cor.png', dpi=220)
    plt.close(fig)


def gerar_ruido_suavizacao(resultados, metricas, exec_json, saida):
    ceu = metricas['falsas_bordas_ceu_y_lt_38pct']
    met = metricas['metricas_realce_recorte']
    paineis = [
        ('5niveis_bin_recorte.png', f"Detecção limpa\n(céu: {ceu['5niveis_limpa_pct']:.2f}% bordas)"),
        ('5niveis_ruidosa_bin_recorte.png', f"Entrada com ruído (σ={exec_json['sigma_ruido']:.0f})\n(céu: {ceu['5niveis_ruidosa_pct']:.2f}% falsas bordas)"),
        ('5niveis_suavizada_bin_recorte.png', f"Gaussiano (σ={exec_json['sigma_gaussiano']}) + Robinson\n(céu: {ceu['5niveis_suavizada_pct']:.3f}% falsas bordas)"),
        ('5niveis_realce_cor_recorte.png', 'Realce sem ruído sintético\n(referência para métricas)'),
        ('5niveis_ruidosa_realce_cor_recorte.png', f"Realce direto com ruído\n(PSNR {met['realce_direto_ruidosa']['psnr_db']:.2f} dB · SSIM {met['realce_direto_ruidosa']['ssim']:.3f})"),
        ('5niveis_suavizada_realce_cor_recorte.png', f"Gaussiano + realce\n(PSNR {met['gaussiano_realce_robinson']['psnr_db']:.2f} dB · SSIM {met['gaussiano_realce_robinson']['ssim']:.3f})"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.6), dpi=220)
    for ax, (arq, titulo) in zip(axes.flat, paineis):
        img = ler_rgb(resultados / arq)
        ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
        ax.set_title(titulo, fontsize=9.0, fontweight='bold', color='#0F172A', pad=6)
        ax.axis('off')
    fig.suptitle('Sensibilidade ao ruído gaussiano e efeito da pré-suavização (3×3, σ = 0,8)', fontsize=11.5, fontweight='bold', y=0.97)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.87, bottom=0.03, hspace=0.34, wspace=0.08)
    fig.savefig(saida / 'fig_ruido_suavizacao.png', dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resultados', type=Path, default=Path('resultados'))
    parser.add_argument('--saida', type=Path, default=Path('resultados/figuras'))
    args = parser.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    exec_json = json.loads((args.resultados / 'execucao.json').read_text(encoding='utf-8'))
    metricas = json.loads((args.resultados / 'metricas.json').read_text(encoding='utf-8'))

    gerar_diagrama_mascaras(args.saida)
    gerar_respostas_4direcoes(args.resultados, args.saida)
    gerar_comparacao_3_vs_5(args.resultados, exec_json, args.saida)
    gerar_mapa_direcoes_e_distribuicao(args.resultados, metricas, args.saida)
    gerar_grafico_distribuicao_direcoes(metricas, args.saida)
    gerar_realce_cinza_cor(args.resultados, exec_json, args.saida)
    gerar_ruido_suavizacao(args.resultados, metricas, exec_json, args.saida)
    print(f'Figuras compostas salvas em: {args.saida.resolve()}')


if __name__ == '__main__':
    main()
