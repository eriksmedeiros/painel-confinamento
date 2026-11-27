# User Stories — Painel Diário do Confinamento

> Histórias de usuário derivadas do [PRD](PRD-Painel-Diario-do-Confinamento.md) e vinculadas aos [Épicos / Milestones](epicos.md) (`E01`–`E05`).
> As histórias usam as três personas do PRD: **Pesquisador/Banca** (rigor experimental e domínio da ementa), **Gestor de Confinamento** (decisão de manejo por lote) e **Leitor/Recrutador** (demonstração interativa e reprodutibilidade).

---

## E01 — Auditoria de Dados, Pré-processamento Local e Splits

### US00 — Preparação do Ambiente GPU no Google Colab e MLflow no Google Drive (`RNF01`, `RNF03`, `RNF04`)
**Como** Pesquisador/Banca,
**Quero** configurar o ambiente de execução com GPU no Google Colab (Google AI Pro) integrado ao Google Drive e ao MLflow em modo *file store* (`mlflow/mlruns/`),
**Para** treinar os Módulos A e B com reprodutibilidade, persistência contra quedas de sessão e rastreamento centralizado de métricas e artefatos leves.

**Critérios de Aceite:**
- [ ] Sessão GPU (L4/A100) no Google Colab validada (`torch.cuda.is_available() == True`) e dependências do `requirements.txt` instaladas.
- [ ] Estrutura `/content/drive/MyDrive/painel-confinamento/{dados,checkpoints,mlflow/mlruns}` criada no Google Drive.
- [ ] Run de teste via `train/tracking.py` registra parâmetros, tags (`seed`, `split`, `git_commit`, `environment`) e métricas no Google Drive sem erro.
- [ ] Callback nativo de MLflow do `ultralytics` validado com `configure_ultralytics_mlflow()`.

### US01 — Auditoria e Splits Sem Vazamento (`RNF01`, `RNF02`, `RNF07`)
**Como** Pesquisador/Banca,
**Quero** que os datasets FBSI e CVB sejam auditados e divididos em treino/validação/teste (70/15/15) salvos em arquivo, separados por clipe (CVB) e fazenda/dia (FBSI),
**Para** garantir que nenhuma métrica seja inflada por vazamento de dados entre conjuntos.

**Critérios de Aceite:**
- [ ] Script de auditoria documenta colunas/metadados do `annotations.csv` (FBSI) e estrutura JSON de caixas/IDs (CVB).
- [ ] Arquivos de split (`train`, `val`, `test`) gerados com `set_seed(42)` e versionados no repositório.
- [ ] Teste automatizado comprova interseção zero de clipes (CVB) e grupos/imagens (FBSI) entre splits.

### US02 — Pré-processamento Local do CVB em Passada Única (`RNF03`, `RNF04`)
**Como** Pesquisador/Banca,
**Quero** subamostrar localmente o CVB de 30 para 5 fps, redimensionar para 640 px (JPEG), converter anotações para formato YOLO e empacotar o derivado,
**Para** subir um arquivo único compacto ao Google Drive e descompactar direto no disco local do Colab sem gargalo de I/O.

**Critérios de Aceite:**
- [ ] Script mantém 1 a cada 6 quadros (30 → 5 fps) e salva JPEG com lado maior = 640 px.
- [ ] Conversor de anotações gera labels YOLO normalizadas em `[0, 1]`, validadas com `pytest`.
- [ ] Derivado compactado disponível no Google Drive; se o upload local atrasar, usar subconjunto estratificado de ~250 clipes mantendo os splits da US01.

---

## E02 — Módulo A: Leitura de Cocho (Escala SDSU)

### US03 — Classificação de Escore de Cocho com CNN e Regressão Ordinal (`RF01`)
**Como** Gestor de Confinamento,
**Quero** que uma foto de cocho seja classificada em um dos 6 escores SDSU (`0`, `0.5`, `1`, `2`, `3`, `4`) com a respectiva confiança,
**Para** padronizar a leitura diária de sobra de ração sem depender de avaliação visual subjetiva.

**Critérios de Aceite:**
- [ ] Modelo CNN (`timm`) treinado no split de treino do FBSI com *data augmentation* (rotação, brilho, recorte).
- [ ] Comparação na validação entre classificação multiclasse padrão e regressão ordinal (reportando Acurácia, F1-macro e MAE ordinal).
- [ ] Inferência retorna `(escore_previsto, confianca)`.

### US04 — Tradução de Escore em Ação de Ajuste e Impacto Prático (`RF02`)
**Como** Gestor de Confinamento,
**Quero** que o escore SDSU previsto seja traduzido automaticamente em uma recomendação de manejo (`reduzir`, `manter` ou `aumentar` ração),
**Para** saber rapidamente como ajustar o trato do lote e medir quantos erros do modelo mudariam a decisão prática.

**Critérios de Aceite:**
- [ ] Regra de mapeamento: `0` → `aumentar`; `0.5` → `manter`; `1`, `2`, `3`, `4` → `reduzir`.
- [ ] Testes unitários cobrem os 6 escores e rejeitam valores fora da escala.
- [ ] Validação reporta % de erros de classificação que alteram a ação recomendada.

### US05 — *(Cortada para 2 semanas / Trabalho Futuro)* Comparação com Vision Transformer no Cocho (`RF07`)
> Removida do escopo de 2 semanas para priorizar a entrega ponta a ponta dos Módulos A e B (o tópico Transformer permanece coberto na US07).

---

## E03 — Módulo B: Comportamento no Curral (Vídeo)

### US06 — Detecção, Rastreamento e Extração Única de Features (`RF03`, `RF04`, `RNF04`)
**Como** Pesquisador/Banca,
**Quero** detectar e rastrear bovinos nos clipes do CVB com YOLO e salvar caixas, IDs e vetores de features em `.npy`/`.parquet`,
**Para** que os modelos temporais treinem em minutos sobre features pré-extraídas sem reprocessar vídeo.

**Critérios de Aceite:**
- [ ] YOLO (`ultralytics`) ajustado no CVB com mAP50 reportado na validação.
- [ ] IDs por animal mantidos ao longo do clipe (via ID nativo ou ByteTrack/BoT-SORT).
- [ ] Artefatos `.npy`/`.parquet` gerados uma única vez para `train`, `val` e `test` dentro do orçamento de ≤ ~3 h de GPU (`RNF03`).

### US07 — Reconhecimento Temporal de Comportamento: LSTM/GRU vs Transformer (`RF05`, `RF06`)
**Como** Pesquisador/Banca,
**Quero** treinar e comparar um modelo recorrente (LSTM/GRU) e um encoder Transformer temporal sobre as mesmas features extraídas (5 fps e janela fixa),
**Para** avaliar F1-macro geral e por comportamento com rigor experimental dentro do prazo de 2 semanas.

**Critérios de Aceite:**
- [ ] LSTM/GRU e Transformer treinados sobre os mesmos artefatos da US06, com mesma seed e splits, registrados no MLflow.
- [ ] Tabela na validação com F1-macro geral e F1 por comportamento para ambos os modelos.
- [ ] *(Nota de escopo)* Grade extensa de ablação de fps e tamanho de janela cortada para trabalho futuro.

---

## E04 — Painel Diário Simulado e Regras de Alerta

### US08 — Dia Simulado por Lote e Motor de Alertas (`RF08`, `RF09`, `RF10`, `RNF06`)
**Como** Gestor de Confinamento,
**Quero** visualizar um painel com um "dia simulado" por lote mostrando escore de cocho por trato, minutos de atividade alimentar por hora e alertas automáticos,
**Para** identificar desvios de alimentação (ex.: sobra alta + baixa atividade no cocho) em uma única tela.

**Critérios de Aceite:**
- [ ] Dia simulado montado só com amostras do conjunto de teste (`RF08`).
- [ ] Regras de alerta (ex.: escore ≥ 2 + atividade alimentar abaixo do limiar) com testes unitários (`RF10`).
- [ ] Painel exibe aviso visível de dia simulado com dados de domínios diferentes (`RNF06`).

### US09 — *(Cortada para 2 semanas / Trabalho Futuro)* Demo Interativa de Foto de Cocho e Vídeo Anotado (`RF11`, `RF12`, `RNF05`)
> Removida do escopo de 2 semanas; o painel foca no dia simulado pré-computado (`US08`).

---

## E05 — Avaliação Final, Deploy e Entrega Acadêmica

### US10 — Avaliação Final no Conjunto de Teste, Deploy e Relatório (`RNF02`, *DoD*)
**Como** Pesquisador/Banca e Leitor/Recrutador,
**Quero** consultar as tabelas finais no conjunto de teste (tocado uma única vez), acessar o painel no Hugging Face Spaces e ler o relatório/post com limitações explícitas,
**Para** validar o rigor experimental e reproduzir o projeto ponta a ponta.

**Critérios de Aceite:**
- [ ] Conjunto de teste avaliado uma única vez com modelos congelados dos Épicos E02 e E03 (`RNF02`).
- [ ] Painel publicado e funcional no Hugging Face Spaces (CPU gratuita).
- [ ] `README.md` atualizado com instruções completas de reprodução e relatório/post publicado.
