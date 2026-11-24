# User Stories — Painel Diário do Confinamento

> Histórias de usuário derivadas do [PRD](PRD-Painel-Diario-do-Confinamento.md) e vinculadas aos [Épicos / Milestones](epicos.md) (`E01`–`E05`).
> As histórias usam as três personas do PRD: **Pesquisador/Banca** (rigor experimental e domínio da ementa), **Gestor de Confinamento** (decisão de manejo por lote) e **Leitor/Recrutador** (demonstração interativa e reprodutibilidade).

---

## E01 — Auditoria de Dados, Pré-processamento Local e Splits

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
**Para** caber no Google Drive gratuito (15 GB) e descompactar direto no disco local do Colab/Kaggle sem gargalo de I/O.

**Critérios de Aceite:**
- [ ] Script mantém 1 a cada 6 quadros (30 → 5 fps) e salva JPEG com lado maior = 640 px.
- [ ] Conversor de anotações gera labels YOLO normalizadas em `[0, 1]`, validadas com `pytest`.
- [ ] Derivado compactado ≤ 15 GB; se exceder, usar subconjunto de 250 clipes mantendo os splits da US01.

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

### US05 — *(Desejável)* Comparação com Vision Transformer no Cocho (`RF07`)
**Como** Pesquisador/Banca,
**Quero** comparar uma variante ViT (`timm`) contra a CNN nos mesmos splits do FBSI,
**Para** avaliar o comportamento de arquiteturas baseadas em atenção com ~1.511 imagens.

**Critérios de Aceite:**
- [ ] Treino do ViT executado com mesma seed e splits da US03 (sujeito ao Gate E02).
- [ ] Tabela comparativa CNN vs ViT gerada na validação.

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
**Quero** treinar e comparar um modelo recorrente (LSTM/GRU) e um encoder Transformer temporal sobre as mesmas features extraídas,
**Para** avaliar F1-macro por comportamento com ablação de taxa de quadros e tamanho de janela.

**Critérios de Aceite:**
- [ ] LSTM/GRU e Transformer treinados sobre os mesmos artefatos da US06, com mesma seed e splits.
- [ ] Tabela na validação com F1-macro geral e F1 por comportamento para ambos os modelos.
- [ ] Ao menos uma ablação de tamanho de janela ou taxa de quadros reportada na validação.

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

### US09 — *(Desejável)* Demo Interativa de Foto de Cocho e Vídeo Anotado (`RF11`, `RF12`, `RNF05`)
**Como** Leitor/Recrutador,
**Quero** enviar uma foto de cocho para receber o escore em até ~2 s em CPU e assistir a um clipe pré-processado com caixas, IDs e comportamentos,
**Para** testar o funcionamento do sistema na prática direto no navegador.

**Critérios de Aceite:**
- [ ] Upload de imagem retorna escore SDSU, confiança e ação recomendada em ≤ ~2 s em CPU (`RNF05`).
- [ ] Player exibe clipe pré-renderizado com bounding boxes, IDs e rótulo de comportamento.

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
