# Épicos do Projeto — Painel Diário do Confinamento

> Decomposição do [PRD](PRD-Painel-Diario-do-Confinamento.md) em 5 épicos sequenciais com portões de passagem (*gates*), alinhados ao cronograma de 4 semanas.

---

## Visão Geral dos Épicos

| ID | Épico | Semana Alvo | Requisitos Cobertos |
| --- | --- | --- | --- |
| **E01** | Auditoria de Dados, Pré-processamento Local e Splits | Semana 1 | `RNF01`, `RNF02`, `RNF04`, `RNF07` |
| **E02** | Módulo A — Leitura de Cocho (Escala SDSU) | Semana 1–2 | `RF01`, `RF02`, `RF07` (Desejável), `RF13`/`RF14` (Opcionais) |
| **E03** | Módulo B — Comportamento no Curral (Vídeo) | Semana 2–3 | `RF03`, `RF04`, `RF05`, `RF06`, `RNF03`, `RNF04` |
| **E04** | Painel Diário Simulado e Regras de Alerta | Semana 3–4 | `RF08`, `RF09`, `RF10`, `RF11`/`RF12` (Desejáveis), `RNF05`, `RNF06` |
| **E05** | Avaliação Final no Teste, Deploy e Relatório | Semana 4 | Critérios de Aceite (*Definition of Done*) |

---

## E01 — Auditoria de Dados, Pré-processamento Local e Splits

**Objetivo:** Garantir divisões treino/validação/teste (70/15/15) livres de vazamento e reduzir localmente o volume do CVB (14,26 GB) em passada única para caber no Google Drive/Colab sem gargalo de I/O.

**Entregas principais:**
- [ ] Auditar `annotations.csv` do FBSI (verificar metadado de fazenda/data e licença).
- [ ] Auditar JSONs do CVB (verificar ID persistente por animal e formato das bounding boxes).
- [ ] Gerar e salvar split 70/15/15 do FBSI (estratificado por escore; agrupado por fazenda/dia se disponível).
- [ ] Gerar e salvar split 70/15/15 do CVB separado estritamente por clipe.
- [ ] Pipeline local de passada única do CVB: subamostrar de 30 para 5 fps (1 a cada 6 quadros), redimensionar (lado maior = 640 px, recomprimir JPEG) e converter anotações para formato YOLO já separado por split + testes unitários do conversor.
- [ ] Empacotar o derivado do CVB em arquivo único compactado para upload ao Drive/Kaggle e descompactação no disco local da sessão (ou aplicar contingência de subconjunto estratificado de ~250 clipes se ainda exceder a cota).

**Portão de Passagem (Gate E01):**
- [ ] Splits FBSI e CVB salvos, sem interseção de clipe/grupo (verificado por teste).
- [ ] Conversor YOLO com testes passando.
- [ ] Pacote derivado do CVB cabe no Drive gratuito; senão, subconjunto de ~250 clipes registrado como limitação.

---

## E02 — Módulo A: Leitura de Cocho (Escala SDSU)

**Objetivo:** Classificar fotos de cocho nos 6 escores SDSU (`0`, `0.5`, `1`, `2`, `3`, `4`) e traduzir o escore em recomendação de manejo (`reduzir`, `manter`, `aumentar`).

**Entregas principais:**
- [ ] Pipeline de dataset e *data augmentation* (rotação, brilho, recorte) para o FBSI em `data/`.
- [ ] Treino de baseline CNN via `timm` (classificação vs regressão ordinal) em `models/` e `train/`.
- [ ] Regra de tradução escore SDSU → ação de ajuste de ração (`RF02`) com testes unitários.
- [ ] Cálculo de Acurácia, F1-macro, MAE ordinal e % de erros que alteram a ação de manejo (na validação).
- [ ] *(Desejável)* Variante ViT (`timm`) para comparação com a CNN (`RF07`).

**Portão de Passagem (Gate E02):**
- [ ] Baseline CNN com F1-macro ≥ 0,85 na validação e MAE ordinal reportado.
- [ ] Regra escore → ação (`RF02`) com testes passando.
- [ ] ViT (`RF07`): só se a CNN fechar até o fim da Semana 2; senão, cortar e registrar.

---

## E03 — Módulo B: Comportamento no Curral (Vídeo)

**Objetivo:** Detectar e rastrear bovinos nos clipes do CVB e classificar o comportamento ao longo do tempo comparando RNN (LSTM/GRU) e Transformer temporal.

**Entregas principais:**
- [ ] Ajuste fino (*fine-tuning*) do YOLO (`ultralytics`) para detecção de bovinos no CVB e avaliação de mAP50 (`RF03`).
- [ ] Rastreamento por clipe (ByteTrack/BoT-SORT ou IDs nativos do CVB) mantendo ID consistente (`RF04`).
- [ ] Extração única de caixas, IDs e features por animal/quadro salvos em `.npy`/`.parquet` (`RNF04`).
- [ ] Modelo recorrente (LSTM/GRU) sobre as features pré-extraídas (`RF05`).
- [ ] Modelo encoder Transformer temporal sobre as mesmas features (`RF06`).
- [ ] Ablação de taxa de quadros e tamanho de janela temporal na validação.

**Portão de Passagem e Plano de Contingência (Gate E03):**
- [ ] Fim da Semana 2: YOLO ajustado com mAP50 reportado e features extraídas em `.npy`/`.parquet` para todos os splits.
- **Gatilho de simplificação:** rastreamento ou extração incompletos no fim da Semana 2 → CNN por quadro + LSTM, sem rastreamento individual.
- **Gatilho do Plano B:** auditoria do CVB mostra rótulos insuficientes para o objetivo → mesmo pipeline com UR Fall / Le2i.

---

## E04 — Painel Diário Simulado e Regras de Alerta

**Objetivo:** Consolidar as saídas dos Módulos A e B em um "dia simulado" interativo em Gradio, rodando em CPU.

**Entregas principais:**
- [ ] Montagem do "dia simulado" por lote usando exclusivamente amostras reservadas do conjunto de teste (`RF08`).
- [ ] Motor de regras de alerta combinando escore de cocho e atividade alimentar (`RF10`) com testes unitários.
- [ ] Interface Gradio (`app/`) exibindo escore por trato, minutos de atividade alimentar por hora e alertas (`RF09`).
- [ ] Aviso explícito de simulação e diferença de domínio entre FBSI e CVB (`RNF06`).
- [ ] *(Desejáveis)* Upload de foto de cocho com inferência em CPU ≤ ~2 s (`RF11`, `RNF05`) e exibição de clipe anotado (`RF12`).

---

## E05 — Avaliação Final, Deploy e Entrega Acadêmica

**Objetivo:** Executar a avaliação única no conjunto de teste, publicar o painel e documentar os resultados e limitações.

**Entregas principais:**
- [ ] Avaliação final única no conjunto de teste (Módulo A e Módulo B) e geração das tabelas comparativas (`RNF02`).
- [ ] Deploy do painel no Hugging Face Spaces (CPU gratuita).
- [ ] Atualização do `README.md` com instruções completas de reprodução.
- [ ] Redação do relatório acadêmico e post técnico para o blog/portfólio.
