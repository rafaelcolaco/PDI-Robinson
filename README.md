# Filtro Bússola de Robinson

Trabalho da disciplina de **Processamento Digital de Imagens (PDI)** — Bacharelado em Ciência da Computação (IFC - Campus Rio do Sul).

Implementação e avaliação do filtro bússola de Robinson (3 e 5 níveis) para detecção direcional e realce de bordas no domínio espacial.

## Conteúdo

- `artigo_robinson.pdf` — Artigo do trabalho
- `slides_robinson.pdf` — Slides da apresentação
- `codigo_e_exemplos/` — Código em Python/OpenCV, imagem de entrada (`assets/`), resultados (`resultados/`) e dependências (`requirements.txt`)

## Como executar

```bash
cd codigo_e_exemplos
pip install -r requirements.txt
python robinson.py assets/casa_horizonte.jpg --saida resultados
python avaliacao_robinson.py assets/casa_horizonte.jpg --saida resultados
python gerar_figuras_artigo.py --resultados resultados --saida resultados/figuras
```
