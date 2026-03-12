# Laboratório 3 — Implementando o Decoder do Transformer

Disciplina: Tópicos em Inteligência Artificial  
Professor: Dimmy Magalhães  
Instituição: iCEV  

## Descrição

Este projeto implementa, em **Python puro com NumPy**, os blocos matemáticos centrais do **Decoder da arquitetura Transformer**, com foco nos mecanismos fundamentais de geração auto-regressiva.

O objetivo do laboratório é demonstrar, de forma prática e estruturada, como o Decoder:

- impede que o modelo “olhe para o futuro” por meio da **máscara causal**
- integra informações produzidas pelo Encoder através da **Cross-Attention**
- gera texto **um token por vez**, simulando o processo de inferência auto-regressiva

A implementação foi construída de forma modular, respeitando os conceitos matemáticos apresentados em sala e no artigo original do Transformer.

---

## Objetivos de Aprendizagem

Este laboratório trabalha três pontos centrais da arquitetura Decoder:

1. **Dominar a álgebra linear da máscara causal**
2. **Implementar a ponte Encoder-Decoder via Cross-Attention**
3. **Simular o loop de inferência auto-regressivo**

---

## Conteúdos Implementados

### 1. Máscara Causal (Look-Ahead Mask)

Foi implementada a função:

`create_causal_mask(seq_len)`

Essa função gera uma matriz quadrada de dimensão `[seq_len, seq_len]`, onde:

- a diagonal principal e a parte triangular inferior recebem `0`
- a parte triangular superior recebe `-inf`

Essa máscara é adicionada à matriz de scores da atenção antes do Softmax, garantindo que posições futuras tenham probabilidade exatamente zero.

Também foi realizada uma demonstração prática com matrizes fictícias `Q` e `K`, mostrando no console:

- os scores sem máscara
- os scores com máscara
- o resultado do Softmax
- a comprovação de que as probabilidades futuras se tornam `0.0`

---

### 2. Cross-Attention (Encoder-Decoder Attention)

Foi implementada a função:

`cross_attention(encoder_out, decoder_state)`

Nesta etapa:

- `decoder_state` é projetado para formar a **Query (Q)**
- `encoder_out` é projetado para formar as **Keys (K)** e os **Values (V)**

Em seguida, é aplicada a equação de **Scaled Dot-Product Attention** sem máscara causal, pois o Decoder deve poder acessar toda a saída do Encoder.

O projeto usa tensores fictícios com as seguintes dimensões:

- `encoder_output`: `[1, 10, 512]`
- `decoder_state`: `[1, 4, 512]`

---

### 3. Loop de Inferência Auto-Regressivo

Foi implementada a função:

`generate_next_token(current_sequence, encoder_out)`

Essa função simula a passagem da sequência atual pelo Decoder e retorna um vetor de probabilidades sobre um vocabulário fictício de tamanho:

`V = 10000`

Depois disso, foi implementado um laço `while` que:

- chama a função iterativamente
- aplica `argmax` para escolher o próximo token
- adiciona o token gerado à sequência
- interrompe a geração quando o token `<EOS>` é produzido

Ao final, o programa imprime a frase gerada.

---

## Estrutura Geral do Código

O código foi organizado em funções para deixar a implementação mais clara, modular e coerente com os blocos lógicos do Transformer.

Principais funções presentes no arquivo:

- `softmax`
- `relu`
- `layer_norm`
- `positional_encoding`
- `create_causal_mask`
- `scaled_dot_product_attention`
- `linear`
- `embed_tokens`
- `masked_self_attention`
- `cross_attention`
- `feed_forward`
- `decoder_block`
- `prepare_decoder_input`
- `generate_next_token`
- `demonstrate_causal_mask`
- `demonstrate_cross_attention`
- `run_autoregressive_inference`
- `main`

---

## Tecnologias Utilizadas

- Python 3.x
- NumPy

---

## Requisitos

Antes de executar, é necessário ter instalado:

- Python 3.10 ou superior
- pip
- NumPy

---

## Como Rodar o Projeto

### 1. Clone ou baixe o projeto

Se estiver usando Git:

```bash
git clone https://github.com/AndreasCarvalho1701/lab-3-dimmy
cd lab-3-dimmy