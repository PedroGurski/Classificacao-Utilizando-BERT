# Classificação de Sentimentos em Análises de Filmes com BERT

Este projeto implementa um modelo de **Processamento de Linguagem Natural (NLP)** utilizando a arquitetura **BERT (Bidirectional Encoder Representations from Transformers)** para realizar a análise de sentimentos em críticas de filmes do IMDB. O sistema classifica as avaliações como **Positivas** ou **Negativas**.

O projeto foi desenvolvido como parte da disciplina de Tópicos em Aprendizado de Máquina, focando na comparação de desempenho entre diferentes configurações de treinamento utilizando Validação Cruzada (*Cross-Validation*).

---

## Dataset

O conjunto de dados utilizado é o **IMDB Dataset of 50k Movie Reviews**, disponível no Kaggle.

* **Fonte:** [Kaggle IMDB Dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
* **Estrutura:** 50.000 avaliações balanceadas (50% positivas, 50% negativas).
* **Amostragem:** Devido a limitações computacionais, foram realizados experimentos com subconjuntos de 5.000 e 6.500 exemplos.
* **Pré-processamento:**
    * Mapeamento de labels: Positive (1) / Negative (0).
    * Tokenização: `BertTokenizer` com truncamento para 128 tokens.

## Arquitetura e Metodologia

O projeto utiliza o modelo pré-treinado **`bert-base-uncased`** (12 camadas, 768 dimensões) da biblioteca Hugging Face Transformers.

### Estratégia de Validação
Foi implementada uma **Validação Cruzada (K-Fold)** com **$k=3$** para garantir a robustez e consistência dos resultados.

### Configurações dos Experimentos
Foram realizados dois experimentos principais para avaliar o impacto do aumento de dados e épocas de treinamento:

| Parâmetro | Experimento 1 | Experimento 2 (Melhor Desempenho) |
| :--- | :---: | :---: |
| **Amostra** | 5.000 exemplos | 6.500 exemplos |
| **Épocas** | 1 | 2 |
| **Batch Size (Treino)** | 4 | 16 |
| **Otimizador** | AdamW (LR: 5e-5) | AdamW (LR: 5e-5) |
| **Device** | CUDA (GPU) | CUDA (GPU) |

## Instalação e Execução

### Pré-requisitos
O código foi desenvolvido em Python e requer as seguintes bibliotecas:

```bash
pip install torch transformers pandas numpy scikit-learn matplotlib kagglehub
