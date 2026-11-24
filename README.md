# Painel Diário do Confinamento

> Protótipo acadêmico que transforma fotos de cocho e vídeos de curral em um resumo diário de manejo alimentar de bovinos, com alertas por regras.

No confinamento, a quantidade de ração é definida por avaliação visual, sujeita a variações e sem considerar o comportamento dos animais, podendo causar desperdícios ou comprometer o ganho de peso. Este projeto trabalha com dois módulos para resolver o problema: classificação da sobra de ração e do comportamento do animal ao longo do tempo. Para isso, utilizou-se métodos de machine learning e visão computacional, como: CNN, YOLO, RNN/Transformer.

> **Nota:** Os datasets utilizados (FBSI e CVB) não são versionados neste repositório.

## Arquitetura do Sistema

- **Módulo A — Cocho:** Classificação da sobra de ração em 6 escores (escala SDSU: `0`, `0.5`, `1`, `2`, `3`, `4`) a partir de fotos.
- **Módulo B — Curral:** Detecção e rastreamento de bovinos em vídeo com classificação temporal de comportamento.
- **Painel:** Consolidação diária simulada por lote com indicadores e alertas de ajuste de ração (`reduzir`, `manter`, `aumentar`).

## Estrutura do Repositório

```text
painel-confinamento/
├── .github/                # Templates de PR e workflow de CI (Ruff)
├── app/                    # Interface Gradio do painel
├── dados/                  # Datasets locais (não versionados no Git)
│   ├── CVB_dataset/
│   └── FBSI_dataset/
├── data/                   # Pré-processamento, splits e datasets PyTorch
├── docs/
│   └── requisitos/         # Especificação e PRD do projeto
├── models/                 # Arquiteturas (Módulo A: Cocho, Módulo B: Curral)
├── tests/                  # Testes unitários
├── train/                  # Scripts de treino, avaliação e reprodutibilidade
├── .gitattributes
├── .gitignore
├── LICENSE
├── pyproject.toml
├── README.md
└── requirements.txt
```

## Documentação

- [PRD — Painel Diário do Confinamento](docs/requisitos/PRD-Painel-Diario-do-Confinamento.md)
- [Épicos e Portões do Projeto](docs/requisitos/epicos.md)
- [User Stories e Critérios de Aceite](docs/requisitos/user-stories.md)
- [AGENTS.md — Regras para Agentes de IA](AGENTS.md)

## Licença

Distribuído sob a licença [MIT](LICENSE).
