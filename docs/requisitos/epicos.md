# Épicos do Projeto — Painel Diário do Confinamento

> Decomposição do [PRD](PRD-Painel-Diario-do-Confinamento.md) em 5 épicos sequenciais com portões de passagem (*gates*), ajustados para o **cronograma de 2 semanas (14 dias)** usando **Google Colab (Google AI Pro)**, **MLflow no Google Drive** e escopo estritamente focado nos requisitos **Obrigatórios**.

---

## Visão Geral dos Épicos (14 dias)

| ID | Épico | Dias Alvo | Requisitos Cobertos |
| --- | --- | --- | --- |
| **E01** | Auditoria de Dados, Pré-processamento Local e Splits | Dias 1–2 | `RNF01`, `RNF02`, `RNF04`, `RNF07` |
| **E02** | Módulo A — Leitura de Cocho (Escala SDSU) | Dias 3–4 | `RF01`, `RF02` *(cortados: `RF07`, `RF13`, `RF14`)* |
| **E03** | Módulo B — Comportamento no Curral (Vídeo) | Dias 5–10 | `RF03`, `RF04`, `RF05`, `RF06`, `RNF03`, `RNF04` |
| **E04** | Painel Diário Simulado, Alertas e Teste Final | Dias 11–12 | `RF08`, `RF09`, `RF10`, `RNF02`, `RNF05`, `RNF06` *(cortados: `RF11`, `RF12`)* |
| **E05** | Deploy, Reprodutibilidade e Entrega Acadêmica | Dias 13–14 | Critérios de Aceite (*Definition of Done*) |

---

## E01 — Auditoria de Dados, Pré-processamento Local e Splits (Dias 1–2)

**Objetivo:** Garantir divisões treino/validação/teste (70/15/15) livres de vazamento e reduzir localmente o volume do CVB (14,26 GB) em passada única para upload rápido ao Google Drive e descompactação no disco local do Colab.

**Entregas principais:**
- [ ] Preparar ambiente de execução com GPU no Google Colab (Google AI Pro), montar o Google Drive, criar a estrutura `painel-confinamento/{dados,checkpoints,mlflow/mlruns}` e validar GPU + `train/tracking.py` (MLflow *file store*).
- [ ] Auditar `annotations.csv` do FBSI (verificar metadado de fazenda/data e licença).
- [ ] Auditar JSONs do CVB (verificar ID persistente por animal e formato das bounding boxes).
- [ ] Gerar e salvar split 70/15/15 do FBSI (estratificado por escore; agrupado por fazenda/dia se disponível).
- [ ] Gerar e salvar split 70/15/15 do CVB separado estritamente por clipe.
- [ ] Pipeline local de passada única do CVB: subamostrar de 30 para 5 fps (1 a cada 6 quadros), redimensionar (lado maior = 640 px, recomprimir JPEG) e converter anotações para formato YOLO já separado por split + testes unitários do conversor.
- [ ] Empacotar o derivado do CVB em arquivo único compactado e subir para o Google Drive (ou usar subconjunto estratificado de ~250 clipes caso o upload local demore demais).

**Portão de Passagem (Gate E01 — fim do Dia 2):**
- [ ] Ambiente Colab (GPU L4/A100) + Google Drive + MLflow (`mlflow/mlruns/`) validados com run de teste.
- [ ] Splits FBSI e CVB salvos, sem interseção de clipe/grupo (verificado por teste).
- [ ] Conversor YOLO com testes passando.
- [ ] Pacote derivado do CVB disponível no Google Drive para uso no Colab.

---

## E02 — Módulo A: Leitura de Cocho — Escala SDSU (Dias 3–4)

**Objetivo:** Classificar fotos de cocho nos 6 escores SDSU (`0`, `0.5`, `1`, `2`, `3`, `4`) com 1 backbone CNN (`timm`) comparando classificação simples vs. regressão ordinal, e traduzir o escore em recomendação de manejo (`reduzir`, `manter`, `aumentar`).

**Cortes aplicados para 2 semanas:** `RF07` (ViT), `RF13` (Grad-CAM) e `RF14` (fotos de campo) ficam fora deste ciclo; o tópico Transformer é coberto no Módulo B (`RF06`).

**Entregas principais:**
- [ ] Pipeline de dataset e *data augmentation* (rotação, brilho, recorte) para o FBSI em `data/`.
- [ ] Treino de baseline CNN via `timm` (classificação simples vs. regressão ordinal) em `models/` e `train/`, com métricas registradas no MLflow (`train/tracking.py`).
- [ ] Regra de tradução escore SDSU → ação de ajuste de ração (`RF02`) com testes unitários.
- [ ] Cálculo de Acurácia, F1-macro, MAE ordinal e % de erros que alteram a ação de manejo na validação.

**Portão de Passagem (Gate E02 — fim do Dia 4):**
- [ ] Baseline CNN validada com F1-macro, MAE ordinal e % de impacto na ação registrados no MLflow.
- [ ] Regra escore → ação (`RF02`) com testes passando.

---

## E03 — Módulo B: Comportamento no Curral — Vídeo (Dias 5–10)

**Objetivo:** Detectar e rastrear bovinos nos clipes do CVB e classificar o comportamento ao longo do tempo comparando RNN (LSTM/GRU) e Transformer temporal sobre as mesmas features pré-extraídas.

**Cortes aplicados para 2 semanas:** grade extensa de ablação de taxa de quadros e tamanho de janela fica fora (trabalhar com **5 fps** e **1 tamanho de janela fixo**).

**Entregas principais:**
- [ ] *(Dias 5–6)* Ajuste fino (*fine-tuning*) do YOLO (`ultralytics`) para detecção de bovinos no CVB com callback MLflow e avaliação de `mAP50` (`RF03`).
- [ ] *(Dias 6–7)* Rastreamento por clipe (ID nativo do CVB ou ByteTrack) e extração única de caixas, IDs e features por animal/quadro salvos em `.npy`/`.parquet` (`RF04`, `RNF04`).
- [ ] *(Dias 8–9)* Modelo recorrente (LSTM/GRU) sobre as features pré-extraídas (`RF05`), logado no MLflow.
- [ ] *(Dias 9–10)* Modelo encoder Transformer temporal sobre as mesmas features (`RF06`), logado no MLflow.

**Portão de Passagem e Contingência (Gate E03):**
- [ ] **Fim do Dia 7:** YOLO ajustado com `mAP50` reportado e features extraídas em `.npy`/`.parquet` para todos os splits.
  - **Gatilho de simplificação (Dia 6):** se o rastreamento por ID travar ou o JSON do CVB não trouxer ID consistente, extrair features por quadro/detecção sem rastreamento individual e seguir imediatamente para `RF05`/`RF06`.
- [ ] **Fim do Dia 10:** LSTM/GRU e Transformer temporal comparados na validação (F1-macro geral e por comportamento) no MLflow.

---

## E04 — Painel Diário Simulado, Alertas e Avaliação no Teste (Dias 11–12)

**Objetivo:** Executar a avaliação final única no conjunto de teste (`RNF02`) e consolidar as saídas dos Módulos A e B em um painel Gradio enxuto rodando em CPU.

**Cortes aplicados para 2 semanas:** `RF11` (upload interativo de foto) e `RF12` (player de clipe anotado) ficam fora do escopo obrigatório.

**Entregas principais:**
- [ ] Avaliação final única no conjunto de teste (Módulo A e Módulo B) com modelos congelados e exportação das tabelas comparativas via MLflow (`RNF02`).
- [ ] Motor de regras de alerta combinando escore de cocho e atividade alimentar (`RF10`) com testes unitários.
- [ ] Montagem do "dia simulado" por lote usando exclusivamente amostras reservadas do conjunto de teste (`RF08`).
- [ ] Interface Gradio (`app/`) exibindo escore por trato, minutos de atividade alimentar por hora, alertas (`RF09`) e aviso explícito de simulação e diferença de domínio (`RNF06`).

**Portão de Passagem (Gate E04 — fim do Dia 12):**
- [ ] Tabelas do conjunto de teste geradas e congeladas.
- [ ] Motor de alertas com testes passando e painel Gradio funcionando ponta a ponta sem erro.

---

## E05 — Deploy, Reprodutibilidade e Entrega Acadêmica (Dias 13–14)

**Objetivo:** Publicar o painel, revisar reprodutibilidade e finalizar a documentação acadêmica.

**Entregas principais:**
- [ ] Deploy do painel no Hugging Face Spaces (CPU gratuita).
- [ ] Atualização do `README.md` com instruções completas de reprodução.
- [ ] Redação do relatório acadêmico (e/ou post técnico) com tabelas comparativas e limitações explicitadas.
