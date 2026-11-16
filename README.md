# Teste: implementação de uma rede neural já treinada

**Matemática Computacional** · Prof. Dr. Alexsandro O. A.

Propagação direta (*forward pass*) com vetores e matrizes, do zero, em Python.

## Sumário

1. [Objetivo e regras](#objetivo-e-regras)
2. [O problema](#o-problema)
3. [Topologia da rede](#topologia-da-rede)
4. [Fórmulas](#fórmulas)
5. [Matrizes de pesos e vetores de viés](#matrizes-de-pesos-e-vetores-de-viés)
6. [Pseudocódigo](#pseudocódigo)
7. [Exemplo de validação](#exemplo-de-validação)
8. [Quiz](#quiz)
9. [Entrega](#entrega)

## Objetivo e regras

A rede deste teste já foi treinada: você não vai ajustar nenhum peso. Sua tarefa é implementar a **propagação direta**, isto é, transformar uma entrada em uma previsão usando só multiplicação de matriz por vetor, soma de vetores e funções de ativação. Depois de validar a implementação com o exemplo resolvido, use-a para responder ao quiz no sistema da disciplina.

- O teste pode ser feito individualmente ou em dupla.
- Só podem ser usados Python, NumPy e pandas. É proibido usar PyTorch, TensorFlow, Keras, `sklearn` ou qualquer rede neural pronta.
- Arquivos fornecidos:

| Arquivo | Conteúdo |
| --- | --- |
| `rede_template.py` | Template com as funções a implementar e o teste de validação |
| `pesos_rede.json` | Topologia, ativações, pesos, vieses e divisores de normalização |
| `entregas_teste_rede.csv` | 10 entregas com a modalidade real, usadas no quiz |

Com Python instalado, crie e ative um ambiente virtual na pasta do projeto,
instale as dependências e rode:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux (use esta linha no lugar da anterior):
# source .venv/bin/activate
python -m pip install -r requirements.txt
python rede_template.py
```

## O problema

Uma agência dos Correios treinou uma rede neural para sugerir a modalidade de envio de uma encomenda. Cada entrega é descrita por quatro atributos brutos:

| Atributo | Unidade | Divisor de normalização |
| --- | --- | --- |
| `distancia_km` | km | 1000 |
| `peso_kg` | kg | 30 |
| `urgencia` | nota de 0 a 1 | 1 |
| `valor_declarado` | reais | 5000 |

Antes de entrar na rede, cada atributo é dividido pelo seu divisor, o que coloca os valores aproximadamente entre 0 e 1. O resultado é o vetor de entrada $`x \in \mathbb{R}^4`$.

A rede devolve três probabilidades, uma para cada classe, sempre nesta ordem: **PAC** (índice 0), **SEDEX** (índice 1) e **SEDEX 10** (índice 2). A classe prevista é a de maior probabilidade.

## Topologia da rede

A rede tem 4 camadas de neurônios, ligadas por 3 matrizes de pesos:

| Camada | Neurônios | Ativação | Pesos que chegam | Viés |
| --- | --- | --- | --- | --- |
| Entrada | 4 | — | — | — |
| Oculta 1 | 6 | ReLU | $`W^{(1)}`$, 6 × 4 | $`b^{(1)}`$, 6 |
| Oculta 2 | 4 | tanh | $`W^{(2)}`$, 4 × 6 | $`b^{(2)}`$, 4 |
| Saída | 3 | softmax | $`W^{(3)}`$, 3 × 4 | $`b^{(3)}`$, 3 |

Convenção: os vetores são **colunas**, e a linha $`i`$ de cada matriz $`W^{(l)}`$ contém os pesos que **chegam** ao neurônio $`i`$ da camada $`l`$.

## Fórmulas

Em cada camada, primeiro calculamos a pré-ativação $`z`$ e depois aplicamos a função de ativação:

$$z^{(1)} = W^{(1)} x + b^{(1)} \qquad h^{(1)} = \mathrm{ReLU}(z^{(1)})$$

$$z^{(2)} = W^{(2)} h^{(1)} + b^{(2)} \qquad h^{(2)} = \tanh(z^{(2)})$$

$$z^{(3)} = W^{(3)} h^{(2)} + b^{(3)} \qquad p = \mathrm{softmax}(z^{(3)})$$

As funções de ativação, aplicadas elemento a elemento:

$$\mathrm{ReLU}(z) = \max(0, z)$$

$$\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$

$$\mathrm{softmax}(z)_i = \frac{e^{z_i - m}}{\sum_{j} e^{z_j - m}}, \quad m = \max_j z_j$$

Subtrair $`m`$ não muda o resultado da softmax, mas evita *overflow* quando algum $`z_i`$ é grande. As probabilidades somam 1, e a classe prevista é o índice da maior delas.

## Matrizes de pesos e vetores de viés

Os mesmos valores estão em `pesos_rede.json`. Nas tabelas, cada linha é um neurônio da camada de destino, e cada coluna é um neurônio da camada de origem (decimais com ponto, como no código).

**Camada oculta 1:** $`W^{(1)}`$ (6 × 4) e $`b^{(1)}`$.

| Neurônio | distância | peso | urgência | valor | viés $`b^{(1)}`$ |
| --- | --- | --- | --- | --- | --- |
| 1 | -3.98 | -1.05 | 2.12 | 0.85 | 3.18 |
| 2 | -1.02 | -0.08 | -0.09 | 1.01 | 0.07 |
| 3 | 2.11 | -0.86 | 1.83 | 0.78 | -0.56 |
| 4 | -0.87 | -1.37 | 0.47 | -1.38 | -0.06 |
| 5 | -0.07 | 0.37 | -0.85 | -0.41 | -0.12 |
| 6 | -3.19 | 2.14 | -5.26 | -1.14 | 5.19 |

**Camada oculta 2:** $`W^{(2)}`$ (4 × 6) e $`b^{(2)}`$.

| Neurônio | $`h_1`$ | $`h_2`$ | $`h_3`$ | $`h_4`$ | $`h_5`$ | $`h_6`$ | viés $`b^{(2)}`$ |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.76 | -0.80 | 0.96 | 0.76 | 0.13 | -1.53 | 0.33 |
| 2 | -1.37 | 0.29 | -0.99 | 0.40 | 0.08 | 2.07 | 0.05 |
| 3 | -1.59 | -0.70 | 1.67 | 0.25 | 0.69 | 4.23 | -1.08 |
| 4 | 2.18 | 1.28 | -1.64 | -0.63 | -0.84 | -0.61 | -0.27 |

**Camada de saída:** $`W^{(3)}`$ (3 × 4) e $`b^{(3)}`$.

| Neurônio | $`h_1`$ | $`h_2`$ | $`h_3`$ | $`h_4`$ | viés $`b^{(3)}`$ |
| --- | --- | --- | --- | --- | --- |
| PAC | -1.78 | 3.33 | 0.82 | 0.12 | -0.22 |
| SEDEX | 1.83 | -0.78 | 1.07 | -0.82 | 1.09 |
| SEDEX 10 | -0.33 | -2.65 | -1.90 | 1.58 | -0.87 |

## Pseudocódigo

```text
FUNÇÃO PROPAGAR(rede, x)
    z1 ← W1 · x  + b1          # vetor de 6 posições
    h1 ← ReLU(z1)
    z2 ← W2 · h1 + b2          # vetor de 4 posições
    h2 ← tanh(z2)
    z3 ← W3 · h2 + b3          # vetor de 3 posições (logits)
    p  ← softmax(z3)           # probabilidades, somam 1
    RETORNAR z1, h1, z2, h2, z3, p

FUNÇÃO CLASSIFICAR(rede, entrega_bruta)
    x ← entrega_bruta ÷ divisores  (elemento a elemento)
    p ← PROPAGAR(rede, x).p
    i ← índice do maior valor de p
    RETORNAR classes[i], p[i]
```

## Exemplo de validação

Para $`x = (0.60,\ 0.40,\ 0.50,\ 0.20)`$, que já está normalizado, sua implementação deve produzir os valores abaixo (arredondados). O template confere tudo automaticamente.

| Etapa | Valor |
| --- | --- |
| $`z^{(1)}`$ | (1.602, -0.417, 1.433, -1.171, -0.521, 1.274) |
| $`h^{(1)}`$ | (1.602, 0, 1.433, 0, 0, 1.274) |
| $`z^{(2)}`$ | (2.5760, -0.9262, 4.1550, 0.0951) |
| $`h^{(2)}`$ | (0.9885, -0.7288, 0.9995, 0.0948) |
| $`z^{(3)}`$ | (-3.5755, 4.4592, -1.0141) |
| $`p`$ | (0.0003, 0.9955, 0.0042) |
| Classe prevista | SEDEX, com 99.55% |

Repare que a ReLU zerou três neurônios da primeira camada oculta: os que tinham pré-ativação negativa.

## Quiz

As 10 perguntas estão no sistema da disciplina. Elas pedem, entre outras coisas: o número de parâmetros da rede; as etapas intermediárias ($`z^{(1)}`$, $`h^{(1)}`$, $`z^{(2)}`$, $`h^{(2)}`$, $`z^{(3)}`$, $`p`$) para uma nova entrada; a classificação de uma entrega a partir dos valores brutos; e o desempenho da rede nas 10 entregas de `entregas_teste_rede.csv`. Mantenha todas as casas decimais nas contas e arredonde só no final.

## Entrega

Os códigos deverão estar em um repositório no GitHub, e o link do repositório deverá ser preenchido na planilha do curso.


## Resultado da implementação

As funções de propagação direta em `rede_template.py` foram implementadas usando NumPy. A execução do arquivo confirmou todos os valores do exemplo de validação.
O programa também imprime a resolução calculada das 10 perguntas do trabalho, incluindo os vetores intermediários, as probabilidades, a acurácia e o gabarito.

- **Total de parâmetros treináveis:** 73
- **Entregas classificadas corretamente:** 8 de 10
- **Acurácia no conjunto de teste:** 80%
- **Entregas classificadas incorretamente:** 5 e 6 (a rede previu PAC, mas a modalidade real é SEDEX)

### Previsões no conjunto de teste

| Entrega | Classe prevista | Probabilidade da classe prevista | Classe real | Correta? |
|---:|---|---:|---|:---:|
| 1 | SEDEX 10 | 95,46% | SEDEX 10 | Sim |
| 2 | PAC | 99,89% | PAC | Sim |
| 3 | SEDEX | 59,43% | SEDEX | Sim |
| 4 | SEDEX | 95,45% | SEDEX | Sim |
| 5 | PAC | 68,57% | SEDEX | Não |
| 6 | PAC | 87,89% | SEDEX | Não |
| 7 | SEDEX 10 | 94,62% | SEDEX 10 | Sim |
| 8 | SEDEX | 99,90% | SEDEX | Sim |
| 9 | PAC | 99,78% | PAC | Sim |
| 10 | PAC | 99,70% | PAC | Sim |
