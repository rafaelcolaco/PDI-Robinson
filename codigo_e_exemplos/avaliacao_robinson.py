"""Auditoria matemática, benchmark 4 vs. 8 máscaras e métricas experimentais do filtro de Robinson.

Uso:
    python avaliacao_robinson.py assets/casa_horizonte.jpg --saida resultados
"""
import argparse
import json
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import skimage
from skimage.metrics import structural_similarity

from robinson import (
    BORDA,
    NOMES_DIRECOES,
    RECORTE_PADRAO,
    VARIANTES,
    expandir_oito_mascaras,
    para_uint8,
    robinson,
    robinson_magnitude,
    robinson_oito_mascaras,
)


def verificar_propriedades(f, limiar):
    """Confere soma nula, resposta constante, degraus sintéticos e equivalência 4 vs. 8."""
    relatorio = {}

    # Degraus sintéticos para validar a convenção de direção do gradiente:
    # R0 = Norte (topo claro, base escura -> borda horizontal)
    degrau_n = np.vstack([np.full((8, 16), 200.0, np.float32), np.full((8, 16), 50.0, np.float32)])
    # R2 = Oeste (esquerda clara, direita escura -> borda vertical)
    degrau_o = np.hstack([np.full((16, 8), 200.0, np.float32), np.full((16, 8), 50.0, np.float32)])
    # R6 = Leste (direita clara, esquerda escura -> borda vertical oposta)
    degrau_l = np.hstack([np.full((16, 8), 50.0, np.float32), np.full((16, 8), 200.0, np.float32)])
    constante = np.full((64, 64), 128.0, np.float32)

    for nome_var, cfg in VARIANTES.items():
        kernels = cfg['kernels']
        fator = cfg['soma_positivos']
        oito = expandir_oito_mascaras(kernels)
        soma_coef = [float(k.sum()) for k in oito]

        mag_const, _, _ = robinson(constante, kernels)
        mag4, dir4, _ = robinson(f, kernels)
        mag_simd = robinson_magnitude(f, kernels)
        mag8, dir8 = robinson_oito_mascaras(f, kernels)

        diff_mag = float(np.max(np.abs(mag4 - mag8)))
        diff_simd = float(np.max(np.abs(mag_simd - mag8)))
        mascara_borda = (mag4 / fator) >= limiar
        concordancia_dir = float(100.0 * np.mean(dir4[mascara_borda] == dir8[mascara_borda]))

        _, dir_n, _ = robinson(degrau_n, kernels)
        _, dir_o, _ = robinson(degrau_o, kernels)
        _, dir_l, _ = robinson(degrau_l, kernels)

        assert all(s == 0.0 for s in soma_coef), f'Soma não nula em {nome_var}: {soma_coef}'
        assert np.max(np.abs(mag_const)) == 0.0, f'Resposta não nula em constante ({nome_var})'
        assert diff_mag == 0.0 and diff_simd == 0.0, f'Divergência 4 vs 8 em {nome_var}'
        assert NOMES_DIRECOES[int(dir_n[7, 8])] == 'N'
        assert NOMES_DIRECOES[int(dir_o[8, 7])] == 'O'
        assert NOMES_DIRECOES[int(dir_l[8, 7])] == 'L'

        relatorio[nome_var] = {
            'soma_coeficientes_oito_mascaras': soma_coef,
            'max_abs_imagem_constante': float(np.max(np.abs(mag_const))),
            'diff_max_magnitude_4_vs_8': diff_mag,
            'diff_max_magnitude_simd_vs_8': diff_simd,
            'concordancia_direcao_em_bordas_pct': concordancia_dir,
            'degrau_topo_claro_direcao': NOMES_DIRECOES[int(dir_n[7, 8])],
            'degrau_esquerda_clara_direcao': NOMES_DIRECOES[int(dir_o[8, 7])],
            'degrau_direita_clara_direcao': NOMES_DIRECOES[int(dir_l[8, 7])],
        }
    return relatorio


def _medir_ms(fn, repeticoes):
    fn()
    tempos = []
    for _ in range(repeticoes):
        t0 = time.perf_counter()
        fn()
        tempos.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(tempos))


def benchmark_4_vs_8(f, repeticoes=20):
    """Mede mediana (ms) para convolução pura, magnitude SIMD e mapa completo."""
    resultados = {}
    for nome_var, cfg in VARIANTES.items():
        kernels = cfg['kernels']
        todos = expandir_oito_mascaras(kernels)

        def conv4():
            return [cv.filter2D(f, cv.CV_32F, k, borderType=BORDA) for k in kernels]

        def conv8():
            return [cv.filter2D(f, cv.CV_32F, k, borderType=BORDA) for k in todos]

        def mag8_simd():
            m = cv.filter2D(f, cv.CV_32F, todos[0], borderType=BORDA)
            for k in todos[1:]:
                cv.max(m, cv.filter2D(f, cv.CV_32F, k, borderType=BORDA), dst=m)
            return m

        c4 = _medir_ms(conv4, repeticoes)
        c8 = _medir_ms(conv8, repeticoes)
        m4 = _medir_ms(lambda: robinson_magnitude(f, kernels), repeticoes)
        m8 = _medir_ms(mag8_simd, repeticoes)
        full4 = _medir_ms(lambda: robinson(f, kernels), repeticoes)
        full8 = _medir_ms(lambda: robinson_oito_mascaras(f, kernels), repeticoes)

        resultados[nome_var] = {
            'convolucao_4_ms': c4,
            'convolucao_8_ms': c8,
            'aceleracao_convolucao_x': float(c8 / c4),
            'magnitude_simd_4_ms': m4,
            'magnitude_simd_8_ms': m8,
            'aceleracao_magnitude_simd_x': float(m8 / m4),
            'completo_dir_4_ms': full4,
            'completo_dir_8_ms': full8,
            'aceleracao_completo_dir_x': float(full8 / full4),
        }
    return resultados


def distribuicao_direcoes(direcao, mag_norm, limiar):
    """Calcula a porcentagem de cada uma das 8 direções entre pixels de borda."""
    mascara = mag_norm >= limiar
    total = int(np.sum(mascara))
    if total == 0:
        return {nome: 0.0 for nome in NOMES_DIRECOES}
    contagens = np.bincount(direcao[mascara], minlength=8)
    return {
        NOMES_DIRECOES[i]: float(100.0 * contagens[i] / total)
        for i in range(8)
    }


def psnr_ssim(referencia, avaliada):
    mse = float(np.mean((avaliada.astype(np.float64) - referencia.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float('inf'), 1.0
    psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
    ssim = float(
        structural_similarity(
            referencia,
            avaliada,
            data_range=255,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    )
    return psnr, ssim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('imagem', nargs='?', default=Path(__file__).parent / 'assets/casa_horizonte.jpg')
    parser.add_argument('--saida', type=Path, default=Path('resultados'))
    parser.add_argument('--limiar', type=float, default=20.0)
    parser.add_argument('--ganho', type=float, default=0.6)
    parser.add_argument('--sigma-gaussiano', type=float, default=0.8)
    parser.add_argument('--sigma-ruido', type=float, default=10.0)
    parser.add_argument('--semente', type=int, default=42)
    parser.add_argument('--repeticoes', type=int, default=20)
    args = parser.parse_args()

    original_bgr = cv.imread(str(args.imagem), cv.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(args.imagem)
    altura, largura = original_bgr.shape[:2]
    x0, y0, x1, y1 = RECORTE_PADRAO
    recorte_sl = np.s_[round(y0 * altura):round(y1 * altura), round(x0 * largura):round(x1 * largura)]

    ycc = cv.cvtColor(original_bgr, cv.COLOR_BGR2YCrCb)
    f = ycc[:, :, 0].astype(np.float32)

    # Faixa superior do céu (y < 38% da altura) onde não há casa, horizonte ou cerca.
    fatia_ceu = np.s_[0:round(0.38 * altura), :]

    verificacao = verificar_propriedades(f, args.limiar)
    bench = benchmark_4_vs_8(f, repeticoes=args.repeticoes)

    distribuicoes = {}
    estatisticas_ceu = {}
    for nome_var, cfg in VARIANTES.items():
        mag, dir_map, _ = robinson(f, cfg['kernels'])
        mag_norm = mag / cfg['soma_positivos']
        distribuicoes[nome_var] = distribuicao_direcoes(dir_map, mag_norm, args.limiar)
        estatisticas_ceu[f'{nome_var}_limpa_pct'] = float(
            100.0 * np.mean(mag_norm[fatia_ceu] >= args.limiar)
        )

    ruido = np.random.default_rng(args.semente).normal(0.0, args.sigma_ruido, f.shape).astype(np.float32)
    ruidosa = np.clip(f + ruido, 0.0, 255.0)
    suavizada = cv.GaussianBlur(ruidosa, (3, 3), args.sigma_gaussiano, borderType=BORDA)

    k5 = VARIANTES['5niveis']['kernels']
    f5 = VARIANTES['5niveis']['soma_positivos']

    mag_limpa, _, _ = robinson(f, k5)
    mag_ruid, _, _ = robinson(ruidosa, k5)
    mag_suav, _, _ = robinson(suavizada, k5)

    mag_limpa_n = mag_limpa / f5
    mag_ruid_n = mag_ruid / f5
    mag_suav_n = mag_suav / f5

    estatisticas_ceu['5niveis_ruidosa_pct'] = float(100.0 * np.mean(mag_ruid_n[fatia_ceu] >= args.limiar))
    estatisticas_ceu['5niveis_suavizada_pct'] = float(100.0 * np.mean(mag_suav_n[fatia_ceu] >= args.limiar))

    ref_realce = para_uint8(f + args.ganho * mag_limpa_n)[recorte_sl]
    candidatos_full = {
        'entrada_limpa': para_uint8(f),
        'entrada_ruidosa': para_uint8(ruidosa),
        'realce_direto_ruidosa': para_uint8(ruidosa + args.ganho * mag_ruid_n),
        'so_gaussiano': para_uint8(suavizada),
        'gaussiano_realce_robinson': para_uint8(suavizada + args.ganho * mag_suav_n),
    }
    metricas_realce = {}
    for nome, img_full in candidatos_full.items():
        std_ceu = float(np.std(img_full[fatia_ceu].astype(np.float64)))
        if nome == 'entrada_limpa':
            metricas_realce[nome] = {'std_ceu': std_ceu}
            continue
        psnr, ssim = psnr_ssim(ref_realce, img_full[recorte_sl])
        metricas_realce[nome] = {'psnr_db': psnr, 'ssim': ssim, 'std_ceu': std_ceu}

    # Erro de ida e volta BGR <-> YCrCb na imagem original e no realce 5 níveis
    ycc_roundtrip = cv.cvtColor(cv.cvtColor(ycc, cv.COLOR_YCrCb2BGR), cv.COLOR_BGR2YCrCb)
    erro_rt_orig = np.abs(ycc.astype(np.int16) - ycc_roundtrip.astype(np.int16))

    ycc_realce = ycc.copy()
    ycc_realce[:, :, 0] = para_uint8(f + args.ganho * mag_limpa_n)
    ycc_realce_rt = cv.cvtColor(cv.cvtColor(ycc_realce, cv.COLOR_YCrCb2BGR), cv.COLOR_BGR2YCrCb)
    erro_rt_cr_cb = np.abs(ycc_realce[:, :, 1:].astype(np.int16) - ycc_realce_rt[:, :, 1:].astype(np.int16))

    relatorio = {
        'versoes': {
            'opencv': cv.__version__,
            'numpy': np.__version__,
            'scikit_image': skimage.__version__,
        },
        'verificacao_matematica': verificacao,
        'benchmark_4_vs_8': bench,
        'distribuicao_direcoes_pct': distribuicoes,
        'falsas_bordas_ceu_y_lt_38pct': estatisticas_ceu,
        'metricas_realce_recorte': metricas_realce,
        'roundtrip_ycrcb': {
            'original_erro_medio': float(np.mean(erro_rt_orig)),
            'original_erro_max': int(np.max(erro_rt_orig)),
            'realce_crcb_erro_medio': float(np.mean(erro_rt_cr_cb)),
            'realce_crcb_erro_max': int(np.max(erro_rt_cr_cb)),
        },
    }

    args.saida.mkdir(parents=True, exist_ok=True)
    (args.saida / 'metricas.json').write_text(
        json.dumps(relatorio, ensure_ascii=False, indent=2), encoding='utf-8'
    )
    print(json.dumps(relatorio, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
