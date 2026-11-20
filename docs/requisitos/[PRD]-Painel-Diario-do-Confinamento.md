# PRD — Painel Diário do Confinamento

Nov 14, 2026 · @Erik Silva de Medeiros

## Resumo

O Painel Diário do Confinamento é um protótipo acadêmico que transforma fotos de cocho e vídeos de curral em um resumo diário da alimentação do gado, com alertas. Será construído por uma pessoa em 4 semanas, com GPU gratuita, como projeto da disciplina de Deep Learning, e publicado no blog/portfólio.

- **Módulo A — Cocho:** classifica a sobra de ração em 6 escores (escala SDSU) a partir de uma foto.
- **Módulo B — Curral:** detecta e rastreia cada animal no vídeo e classifica seu comportamento ao longo do tempo.
- **Painel:** combina os dois num "dia simulado" com indicadores por lote e alertas por regras.

O produto entrega **indicadores de manejo alimentar**, não estimativa de peso nem de ganho de peso.

## Problema e contexto

No confinamento, a quantidade diária de ração é ajustada pela leitura visual do cocho: um tratador avalia a sobra do dia anterior e dá uma nota. A leitura é subjetiva, varia entre avaliadores e não acompanha o comportamento dos animais ao longo do dia.

A alimentação pode chegar a 60%–80% dos custos operacionais do confinamento ([Data in Brief, 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9989516/)). Comida demais gera desperdício e risco de distúrbios digestivos; comida de menos reduz o ganho de peso.

A literatura já mostra que visão computacional consegue classificar o escore de cocho ([Dorea & Cheong, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6898560); [Rezende da Silva et al., 2025](https://doi.org/10.1007/s11760-025-04670-y)) e detectar comportamento alimentar ([Pi et al., 2023](https://www.preprints.org/manuscript/202307.0705)). Falta um protótipo aberto que combine os dois sinais num indicador diário.

## Objetivos e não objetivos

**Objetivos**

1. Classificar o escore de cocho com desempenho comparável ao da literatura no FBSI.
2. Reconhecer o comportamento de cada animal em vídeo, comparando modelo recorrente (LSTM/GRU) e transformer temporal.
3. Medir o impacto prático dos erros: quantos erros de escore levariam a um ajuste errado da ração.
4. Entregar um painel online demonstrável e um post técnico no blog.
5. Cobrir os quatro tópicos da ementa: CNN, YOLO, RNN e transformers.

**Não objetivos**

- Estimar peso ou ganho médio diário (não há dados públicos de Nelore com peso).
- Identificar individualmente animais Nelore entre dias.
- Operar em tempo real ou em uma fazenda real.
- Coletar ou anotar um dataset próprio em larga escala.

## Usuários e casos de uso

| Usuário | O que precisa | Caso de uso principal |
| --- | --- | --- |
| Professor / banca | Ver domínio da ementa e rigor experimental | Lê o relatório e as tabelas comparativas; testa o painel |
| Gestor de confinamento (persona simulada) | Saber, por lote, se a ração está ajustada | Abre o painel do dia, vê escore de cocho, atividade alimentar e alertas |
| Leitor do blog / recrutador | Entender o problema e ver o sistema funcionando | Lê o post e envia uma foto de cocho para a demo |

O gestor é uma persona de referência para guiar o design; não haverá validação com usuários reais, salvo se a visita a um confinamento acontecer (ver Questões em aberto).

## Requisitos funcionais

Tudo marcado como Obrigatório cabe nas 4 semanas; Desejável entra se a semana correspondente fechar no prazo; Opcional é trabalho futuro.

| ID | Módulo | Requisito | Prioridade |
| --- | --- | --- | --- |
| RF01 | A — Cocho | Classificar a foto do cocho em 6 escores SDSU (0, 0,5, 1, 2, 3, 4), com a confiança da previsão | Obrigatório |
| RF02 | A — Cocho | Traduzir o escore em ação de ajuste da ração: reduzir, manter ou aumentar | Obrigatório |
| RF03 | B — Curral | Detectar bovinos em cada quadro de vídeo (YOLO ajustado no CVB) | Obrigatório |
| RF04 | B — Curral | Rastrear cada animal e manter seu ID dentro do clipe | Obrigatório |
| RF05 | B — Curral | Classificar o comportamento de cada animal por janela de tempo com LSTM/GRU | Obrigatório |
| RF06 | B — Curral | Variante com encoder transformer temporal, treinada sobre as mesmas features | Obrigatório |
| RF07 | A — Cocho | Variante ViT para comparar com a CNN | Desejável |
| RF08 | Painel | Montar um "dia simulado" só com amostras de teste (fotos FBSI + clipes CVB) | Obrigatório |
| RF09 | Painel | Exibir por lote: escore por trato e minutos de atividade alimentar por hora | Obrigatório |
| RF10 | Painel | Gerar alertas por regras (ex.: sobra alta + baixa atividade alimentar) | Obrigatório |
| RF11 | Painel | Permitir que o usuário envie uma foto de cocho e receba o escore | Desejável |
| RF12 | Painel | Exibir um clipe com caixas, IDs e comportamento sobrepostos | Desejável |
| RF13 | A — Cocho | Mostrar mapa de atenção (Grad-CAM) da previsão | Opcional |
| RF14 | A — Cocho | Avaliar o modelo em 30 a 50 fotos reais coletadas em campo | Opcional |

## Requisitos não funcionais

| ID | Categoria | Requisito |
| --- | --- | --- |
| RNF01 | Reprodutibilidade | Seeds fixas, divisão treino/validação/teste salva em arquivo, ambiente versionado (`requirements.txt`) |
| RNF02 | Rigor experimental | Separação por clipe (CVB) e por fazenda/dia quando o metadado existir (FBSI); conjunto de teste tocado uma única vez |
| RNF03 | Custo computacional | Todo treino cabe em GPU gratuita (Colab/Kaggle); nenhuma rodada de treino acima de \~3 h |
| RNF04 | Armazenamento | Features e caixas pré-extraídas em `.npy`/`.parquet`; quadros brutos do CVB (14,26 GB) processados uma vez |
| RNF05 | Desempenho da demo | Inferência de uma foto de cocho em CPU em até \~2 s; o vídeo do painel é pré-processado |
| RNF06 | Transparência | O painel informa que o dia é simulado e que os domínios dos datasets são diferentes |
| RNF07 | Licenças | Uso não comercial e atribuição (CVB: CC BY-NC-SA 4.0; FBSI: conferir no Mendeley); código aberto no GitHub |
| RNF08 | Qualidade de código | Estrutura modular (`data/`, `models/`, `train/`, `app/`); código gerado por agentes de IA revisado antes do commit |

## Dados

Os dois datasets já foram baixados e auditados: o CVB tem as pastas `annotations`, `CVB_in_AVA_format` e `raw_frames` (14,26 GB).

| Dataset | Conteúdo | Domínio | Licença | Uso |
| --- | --- | --- | --- | --- |
| [FBSI](https://data.mendeley.com/datasets/p9cr67s6jp) | 1.511 fotos de cocho em 6 escores SDSU, com composição da dieta | Confinamentos de corte no Brasil (UFMG/UFJF) | Conferir no Mendeley | Módulo A |
| [CVB](https://researchdata.edu.au/cvb-a-video-visual-behaviors/2593248) | 502 clipes de 15 s (450 quadros a 30 fps), caixas e 11 comportamentos por animal | 8 bovinos Angus em pasto (CSIRO, Austrália) | CC BY-NC-SA 4.0 | Módulo B |

**Pré-processamento**

1. **FBSI:** redimensionar, aumentar dados (rotação, brilho, recorte) e dividir 70/15/15 de forma estratificada; por fazenda ou dia se o CSV permitir.
2. **CVB:** subamostrar de 30 para 5 fps; converter as anotações para o formato YOLO; dividir por clipe (70/15/15).
3. **Features:** rodar detector + extrator uma vez e salvar caixas, IDs e vetores por animal/quadro; os modelos temporais treinam sobre esses arquivos.

**Pendente:** confirmar no JSON do CVB se há ID persistente por animal e o formato das caixas (pixels ou normalizado).

## Arquitetura

Os dois módulos são independentes e só se encontram no painel; um pode atrasar sem bloquear o outro.

&#91;embedded content: arquitetura · módulo A (cocho), módulo B (curral) e painel\]

O processamento pesado (detecção, rastreamento, extração de features) roda uma vez no Colab/Kaggle. O painel só carrega resultados prontos e faz inferência leve da foto de cocho em CPU.

## Recursos e restrições

| Recurso | Disponível | Restrição / decisão |
| --- | --- | --- |
| Equipe | 1 aluno, em paralelo com trabalho e outras disciplinas | Escopo travado nos requisitos Obrigatórios |
| Prazo | 4 semanas | Marcos semanais com critério de passagem |
| GPU | Google Colab / Kaggle (gratuitos) | Sessões e cotas limitadas: checkpoints frequentes, features pré-extraídas |
| Armazenamento | Google Drive pessoal | CVB processado a 5 fps; guardar só artefatos derivados |
| Stack | Python, PyTorch, Ultralytics (YOLO + rastreador ByteTrack/BoT-SORT), timm (CNN/ViT), Gradio | Bibliotecas maduras e documentadas; nada implementado do zero sem necessidade |
| Hospedagem | Hugging Face Spaces (CPU gratuita) | Demo só com inferência leve; vídeo pré-processado |
| Agentes de IA | Claude Code / assistentes de código | Usados em boilerplate, conversão de dados, scripts e painel; análise e interpretação feitas pelo aluno |
| Versionamento | GitHub | Commits em PT-BR, README com instruções de reprodução |

## Métricas de sucesso e critérios de aceite

As metas abaixo são referências, não promessas: um resultado abaixo da meta, bem analisado, ainda cumpre o objetivo acadêmico.

| Área | Métrica | Meta / referência |
| --- | --- | --- |
| Módulo A | Acurácia e F1-macro no teste | Próximo dos 95,37% do ensemble publicado ([Rezende da Silva et al., 2025](https://doi.org/10.1007/s11760-025-04670-y)) |
| Módulo A | MAE ordinal (erro médio em "degraus" de escore) | Menor na regressão ordinal que na classificação simples |
| Módulo A | % de previsões que mudariam a ação de ajuste | Reportado e discutido |
| Módulo B | mAP50 da detecção de bovinos | Reportado; base para o rastreamento |
| Módulo B | F1-macro por comportamento (LSTM/GRU vs transformer) | Tabela comparativa com ablação de taxa de quadros e tamanho de janela |
| Painel | Funcionamento ponta a ponta | Dia simulado carrega, indicadores e alertas aparecem sem erro |

**Critérios de aceite (Definition of Done)**

- [ ] Todos os requisitos Obrigatórios implementados
- [ ] Resultados no conjunto de teste, com divisão salva e reprodutível
- [ ] Tabelas comparativas CNN vs ViT e LSTM/GRU vs transformer
- [ ] Painel publicado no Hugging Face Spaces
- [ ] Repositório com README de reprodução
- [ ] Relatório e post no blog com limitações explicitadas

## Cronograma e marcos

Cada fase termina num portão: só se avança quando o critério é cumprido, ou aciona-se a alternativa prevista em Riscos.

&#91;embedded content: cronograma · 5 fases e seus portões\]

Datas assumindo início em 16/11/2026 e entrega em 15/12/2026; ajustar quando a data oficial da disciplina for confirmada (ver Questões em aberto).

## Riscos e mitigações

| Risco | Probabilidade | Impacto | Mitigação |
| --- | --- | --- | --- |
| Anotações do CVB sem ID persistente por animal | Média | Médio | Gerar IDs com o rastreador do Ultralytics e conferir uma amostra à mão |
| Módulo B atrasa além da semana 2 | Média | Alto | Simplificar: features por quadro (CNN) + LSTM, sem rastreamento individual |
| CVB inadequado para o objetivo | Baixa | Alto | **Plano B:** mesmo pipeline aplicado à detecção de quedas (UR Fall / Le2i) |
| Cota de GPU gratuita esgota | Média | Médio | Checkpoints frequentes, features pré-extraídas, alternar Colab e Kaggle |
| Vazamento entre treino e teste infla métricas | Média | Alto | Divisão por clipe/fazenda definida no dia 1 e salva em arquivo |
| ViT ou transformer temporal rende pouco com poucos dados | Alta | Baixo | É um resultado válido; analisar e reportar, não esconder |
| Diferença de domínio (pasto/Angus vs confinamento/Nelore) questionada pela banca | Alta | Médio | Declarar como limitação no painel, no relatório e no post |
| Código gerado por IA com erro silencioso | Média | Médio | Testes simples nos conversores de dados; revisão antes de cada commit |

## Questões em aberto

- [ ] Qual é a data de entrega e o formato exigido pela disciplina (relatório, apresentação, código)?
- [ ] O professor aceita a troca do tema apresentado no pitch (MVTec AD)?
- [ ] O JSON do CVB traz ID persistente por animal?
- [ ] O CSV do FBSI identifica fazenda ou data, para separar treino e teste por grupo?
- [ ] Qual é a licença do FBSI?
- [ ] Há acesso a um confinamento (ex.: fazenda-escola) para fotografar cochos reais (RF14)?

## Fontes

- [Feed Bunk Score Images — FBSI (Mendeley)](https://data.mendeley.com/datasets/p9cr67s6jp)
- [Dataset of feed bunk score images of cattle feedlot — Data in Brief, 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9989516/)
- [Feed bunk score images classification using deep transfer learning — Rezende da Silva et al., 2025](https://doi.org/10.1007/s11760-025-04670-y)
- [CVB: A Video Dataset of Cattle Visual Behaviors — dataset (CSIRO)](https://researchdata.edu.au/cvb-a-video-visual-behaviors/2593248)
- [CVB: A Video Dataset of Cattle Visual Behaviors — artigo (arXiv)](https://arxiv.org/abs/2305.16555)
- [A computer vision system for feed bunk management in beef cattle feedlot — Dorea & Cheong, 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6898560)
- [Using Computer Vision's Pixel Segmentation to Detect Beef Cattle Feeding Behavior — Pi et al., 2023](https://www.preprints.org/manuscript/202307.0705)
