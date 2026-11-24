# AGENTS.md — Instruções para Agentes de Codificação

> Guia universal de contexto e restrições para qualquer agente de IA (Claude Code, Codex, Cursor, Antigravity, etc.) trabalhando neste repositório.

## Contexto do Projeto

Protótipo acadêmico de visão computacional para leitura de cocho (escala SDSU, Módulo A) e comportamento bovino em vídeo (Módulo B), consolidados em um painel Gradio.
Consulte a especificação completa em [docs/requisitos/PRD-Painel-Diario-do-Confinamento.md](docs/requisitos/PRD-Painel-Diario-do-Confinamento.md).

## Estrutura Modular (`RNF08`)

- `data/` — Pré-processamento, conversores de anotação e splits salvos em arquivo.
- `models/` — Arquiteturas (Módulo A: CNN/ViT; Módulo B: YOLO + LSTM/GRU/Transformer).
- `train/` — Scripts de treino, validação e reprodutibilidade (`train/seed.py`).
- `app/` — Interface Gradio do painel (inferência leve em CPU).
- `tests/` — Testes unitários (priorizar conversores de dados e regras de alerta).
- `dados/` — Datasets locais (~14,8 GB, **ignorados pelo Git**; nunca copiar ou commitar arquivos daqui).

## Convenções de Código e Git

- **Python 3.11+**, formatado e validado com Ruff (`ruff check .` e `ruff format --check .`).
- **Nunca commitar** na `main` diretamente: crie uma branch (`feat/...`, `fix/...`, `docs/...`, `chore/...`) e abra PR.
- **Commits:** Padrão *Conventional Commits* com descrição em **PT-BR** (ex.: `feat: adiciona split estratificado do FBSI`).
- **Dependências:** Ao adicionar biblioteca nova, atualizar `requirements.txt` (`RNF01`).

## Regras de Rigor Experimental e Revisão (`RNF01`–`RNF08`)

1. **Seed:** chamar `set_seed()` de `train/seed.py` em todo treino/avaliação (`RNF01`).
2. **Splits:** carregar só dos arquivos salvos; separar por clipe (CVB) e fazenda/dia (FBSI) (`RNF02`).
3. **Teste:** usar só na avaliação final; nunca em tuning ou seleção de modelo (`RNF02`).
4. **GPU e Armazenamento:** pré-processar o CVB localmente em passada única (5 fps, lado maior 640 px JPEG, labels YOLO) e subir só o derivado compactado para descompactar no disco local do Colab/Kaggle; treinos ≤ ~3 h sobre features `.npy`/`.parquet` (`RNF03`/`RNF04`).
5. **Reuso:** usar `timm` e `ultralytics`; não reimplementar o que já existe.
