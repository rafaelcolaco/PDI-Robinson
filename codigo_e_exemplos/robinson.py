"""Detecção e realce de bordas com o filtro bússola de Robinson (3 e 5 níveis).

Uso:
    python robinson.py assets/casa_horizonte.jpg --saida resultados
"""
import argparse
import json
from pathlib import Path

import cv2 as cv
import numpy as np

BORDA = cv.BORDER_REFLECT_101
RECORTE_PADRAO = (0.20, 0.35, 0.80, 0.90)

# Convenção adotada: R0..R3 giram 45° no sentido anti-horário a partir do Norte.
# Os pesos positivos indicam o sentido do vetor gradiente (do escuro para o claro).
NOMES_DIRECOES = ('N', 'NO', 'O', 'SO', 'S', 'SE', 'L', 'NE')

# Paleta BGR para exibir as 8 orientações (cores opostas com matizes contrastantes).
CORES_DIRECOES_BGR = np.array([
    [ 60,  76, 231],  # N  - vermelho
    [ 15, 196, 241],  # NO - amarelo
    [113, 204,  46],  # O  - verde
    [188, 188,  26],  # SO - ciano
    [219, 152,  52],  # S  - azul
    [182,  89, 155],  # SE - roxo
    [132,  99, 255],  # L  - rosa
    [ 34, 126, 230],  # NE - laranja
], dtype=np.uint8)

ROBINSON_3 = np.array([
    [[ 1,  1,  1], [ 0,  0,  0], [-1, -1, -1]],
    [[ 1,  1,  0], [ 1,  0, -1], [ 0, -1, -1]],
    [[ 1,  0, -1], [ 1,  0, -1], [ 1,  0, -1]],
    [[ 0, -1, -1], [ 1,  0, -1], [ 1,  1,  0]],
], dtype=np.float32)

ROBINSON_5 = np.array([
    [[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]],
    [[ 2,  1,  0], [ 1,  0, -1], [ 0, -1, -2]],
    [[ 1,  0, -1], [ 2,  0, -2], [ 1,  0, -1]],
    [[ 0, -1, -2], [ 1,  0, -1], [ 2,  1,  0]],
], dtype=np.float32)

VARIANTES = {
    '3niveis': {'kernels': ROBINSON_3, 'soma_positivos': 3.0},
    '5niveis': {'kernels': ROBINSON_5, 'soma_positivos': 4.0},
}


def expandir_oito_mascaras(kernels_base):
    """Expande os 4 kernels base para o conjunto completo de 8 direções."""
    return np.concatenate([kernels_base, -kernels_base], axis=0)


def robinson_magnitude(imagem, kernels_base=ROBINSON_5, border_type=BORDA):
    """Calcula apenas a magnitude das bordas com 4 convoluções e cv.max (caminho rápido)."""
    magnitude = np.abs(cv.filter2D(imagem, cv.CV_32F, kernels_base[0], borderType=border_type))
    for k in kernels_base[1:]:
        cv.max(magnitude, np.abs(cv.filter2D(imagem, cv.CV_32F, k, borderType=border_type)), dst=magnitude)
    return magnitude


def robinson(imagem, kernels_base=ROBINSON_5, border_type=BORDA):
    """Calcula magnitude, direção (0..7) e as 4 respostas base usando simetria.

    Como R_{k+4} = -R_k, basta filtrar com as 4 máscaras base:
      max_{k=0..7} R_k = max_{k=0..3} |R_k|
    e o sinal de R_{k*} decide entre k* e k* + 4.
    """
    respostas = np.empty((len(kernels_base), *imagem.shape), dtype=np.float32)
    for idx, k in enumerate(kernels_base):
        cv.filter2D(imagem, cv.CV_32F, k, dst=respostas[idx], borderType=border_type)
    idx_base = np.argmax(np.abs(respostas), axis=0)
    resp_max = np.take_along_axis(respostas, idx_base[None, ...], axis=0)[0]
    magnitude = np.abs(resp_max)
    direcao = np.where(resp_max >= 0.0, idx_base, idx_base + 4).astype(np.uint8)
    return magnitude, direcao, respostas


def robinson_oito_mascaras(imagem, kernels_base=ROBINSON_5, border_type=BORDA):
    """Versão direta com 8 convoluções, usada apenas para validação e benchmark."""
    todos_kernels = expandir_oito_mascaras(kernels_base)
    respostas = np.stack(
        [cv.filter2D(imagem, cv.CV_32F, k, borderType=border_type) for k in todos_kernels],
        axis=0,
    )
    direcao = np.argmax(respostas, axis=0).astype(np.uint8)
    magnitude = np.take_along_axis(respostas, direcao[None, ...], axis=0)[0]
    return magnitude, direcao


def para_uint8(imagem):
    """Limita a [0, 255], arredonda e converte para uint8."""
    return np.rint(np.clip(imagem, 0.0, 255.0)).astype(np.uint8)


def colorir_direcoes(direcao, mag_norm, limiar):
    """Mapeia direções 0..7 para BGR apenas onde a magnitude atinge o limiar."""
    saida = np.zeros((*direcao.shape, 3), dtype=np.uint8)
    mascara = mag_norm >= limiar
    saida[mascara] = CORES_DIRECOES_BGR[direcao[mascara]]
    return saida


def aplicar_realce_cor(ycc_original, canal_y_uint8):
    """Substitui a componente Y preservando Cr e Cb e converte para BGR."""
    canais = ycc_original.copy()
    canais[:, :, 0] = canal_y_uint8
    return cv.cvtColor(canais, cv.COLOR_YCrCb2BGR)


def salvar(pasta, nome, imagem, recorte):
    """Salva a imagem completa e o recorte em PNG."""
    x0, y0, x1, y1 = recorte
    if not cv.imwrite(str(pasta / f'{nome}.png'), imagem):
        raise OSError(f'Falha ao salvar {nome}.png')
    if not cv.imwrite(str(pasta / f'{nome}_recorte.png'), imagem[y0:y1, x0:x1]):
        raise OSError(f'Falha ao salvar {nome}_recorte.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imagem', nargs='?', default=Path(__file__).parent / 'assets/casa_horizonte.jpg')
    parser.add_argument('--saida', type=Path, default=Path('resultados'))
    parser.add_argument('--limiar', type=float, default=20.0,
                        help='Limiar na escala normalizada [0, 255] para mapa binário e direcional.')
    parser.add_argument('--ganho', type=float, default=0.6,
                        help='Ganho α aplicado à magnitude normalizada no realce.')
    parser.add_argument('--sigma-gaussiano', type=float, default=0.8)
    parser.add_argument('--sigma-ruido', type=float, default=10.0)
    parser.add_argument('--semente', type=int, default=42)
    parser.add_argument('--recorte', nargs=4, type=float, default=RECORTE_PADRAO)
    args = parser.parse_args()

    parametros = (args.limiar, args.ganho, args.sigma_gaussiano, args.sigma_ruido)
    if not all(np.isfinite(v) and v >= 0 for v in parametros) or args.semente < 0:
        parser.error('Use parâmetros finitos e não negativos.')
    x0, y0, x1, y1 = args.recorte
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        parser.error('Use um recorte normalizado entre 0 e 1, com início menor que fim.')

    original_bgr = cv.imread(str(args.imagem), cv.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(args.imagem)
    altura, largura = original_bgr.shape[:2]
    recorte = (round(x0 * largura), round(y0 * altura),
               round(x1 * largura), round(y1 * altura))
    if recorte[2] <= recorte[0] or recorte[3] <= recorte[1]:
        parser.error('O recorte precisa conter pelo menos um pixel.')
    args.saida.mkdir(parents=True, exist_ok=True)

    ycc = cv.cvtColor(original_bgr, cv.COLOR_BGR2YCrCb)
    f = ycc[:, :, 0].astype(np.float32)

    rng = np.random.default_rng(args.semente)
    ruido = rng.normal(0.0, args.sigma_ruido, f.shape).astype(np.float32)
    ruidosa = np.clip(f + ruido, 0.0, 255.0)
    suavizada = cv.GaussianBlur(ruidosa, (3, 3), args.sigma_gaussiano, borderType=BORDA)

    salvar(args.saida, 'original_cinza', para_uint8(f), recorte)
    salvar(args.saida, 'original_cor', original_bgr, recorte)
    salvar(args.saida, 'ruidosa_cinza', para_uint8(ruidosa), recorte)
    salvar(args.saida, 'suavizada_cinza', para_uint8(suavizada), recorte)
    salvar(args.saida, 'ruidosa_cor', aplicar_realce_cor(ycc, para_uint8(ruidosa)), recorte)
    salvar(args.saida, 'suavizada_cor', aplicar_realce_cor(ycc, para_uint8(suavizada)), recorte)

    estatisticas = {}
    for nome_var, cfg in VARIANTES.items():
        kernels = cfg['kernels']
        fator = cfg['soma_positivos']

        mag, dir_map, resp_base = robinson(f, kernels)
        mag_norm = mag / fator
        binaria = np.where(mag_norm >= args.limiar, 255, 0).astype(np.uint8)
        dir_bgr = colorir_direcoes(dir_map, mag_norm, args.limiar)
        realce_y = para_uint8(f + args.ganho * mag_norm)
        realce_bgr = aplicar_realce_cor(ycc, realce_y)

        salvar(args.saida, f'{nome_var}_mag', para_uint8(mag_norm), recorte)
        salvar(args.saida, f'{nome_var}_bin', binaria, recorte)
        salvar(args.saida, f'{nome_var}_dir', dir_bgr, recorte)
        salvar(args.saida, f'{nome_var}_realce_cinza', realce_y, recorte)
        salvar(args.saida, f'{nome_var}_realce_cor', realce_bgr, recorte)

        # Respostas base para inspecionar seletividade direcional (ex.: N/S e L/O).
        for idx in range(4):
            vis_dir = para_uint8(128.0 + 0.5 * (resp_base[idx] / fator))
            salvar(args.saida, f'{nome_var}_resp_R{idx}', vis_dir, recorte)

        estatisticas[nome_var] = {
            'mag_bruta_max': float(mag.max()),
            'mag_norm_max': float(mag_norm.max()),
            'pixels_borda_pct': float(100.0 * np.mean(mag_norm >= args.limiar)),
        }

    # Ensaio de ruído e pré-suavização usando a variante principal de 5 níveis.
    k5 = VARIANTES['5niveis']['kernels']
    f5 = VARIANTES['5niveis']['soma_positivos']
    for rotulo, entrada in (('ruidosa', ruidosa), ('suavizada', suavizada)):
        mag_r, dir_r, _ = robinson(entrada, k5)
        mag_r_norm = mag_r / f5
        bin_r = np.where(mag_r_norm >= args.limiar, 255, 0).astype(np.uint8)
        dir_r_bgr = colorir_direcoes(dir_r, mag_r_norm, args.limiar)
        realce_r_y = para_uint8(entrada + args.ganho * mag_r_norm)
        realce_r_bgr = aplicar_realce_cor(ycc, realce_r_y)

        salvar(args.saida, f'5niveis_{rotulo}_mag', para_uint8(mag_r_norm), recorte)
        salvar(args.saida, f'5niveis_{rotulo}_bin', bin_r, recorte)
        salvar(args.saida, f'5niveis_{rotulo}_dir', dir_r_bgr, recorte)
        salvar(args.saida, f'5niveis_{rotulo}_realce_cinza', realce_r_y, recorte)
        salvar(args.saida, f'5niveis_{rotulo}_realce_cor', realce_r_bgr, recorte)

    registro = {
        'imagem': Path(args.imagem).name,
        'dimensoes': [altura, largura],
        'borda': 'BORDER_REFLECT_101',
        'limiar': args.limiar,
        'ganho': args.ganho,
        'gaussiano': [3, 3],
        'sigma_gaussiano': args.sigma_gaussiano,
        'sigma_ruido': args.sigma_ruido,
        'semente': args.semente,
        'recorte_xyxy': list(recorte),
        'estatisticas': estatisticas,
    }
    (args.saida / 'execucao.json').write_text(
        json.dumps(registro, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(f'Resultados salvos em: {args.saida.resolve()}')


if __name__ == '__main__':
    main()
