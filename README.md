# Implementação de Rede Neural para Classificação de Entregas

## Sobre o projeto

Trabalho acadêmico desenvolvido em Python para implementar a propagação direta (*forward pass*) de uma rede neural artificial responsável por classificar modalidades de entrega: PAC, SEDEX e SEDEX 10.

## Objetivos

- Implementar funções de ativação ReLU, tanh e softmax.
- Normalizar os atributos de entrada.
- Realizar os cálculos das camadas da rede neural.
- Utilizar pesos e vieses previamente fornecidos.
- Classificar entregas e avaliar o desempenho do modelo.

## Tecnologias utilizadas

- Python
- NumPy
- pandas
- Arquivos JSON e CSV

## Arquitetura da rede

A rede possui a topologia 4 → 6 → 4 → 3, com ReLU na primeira camada oculta, tanh na segunda camada oculta e softmax na camada de saída.

## Como executar

Instale as dependências:

```bash
pip install numpy pandas
```

Execute o programa:

```bash
python rede_template.py
```

Mantenha os arquivos de pesos e dados nos caminhos esperados pelo código.

## Resultados

A avaliação utiliza o arquivo de teste fornecido na atividade. A acurácia e as classificações devem ser conferidas pela execução do programa com os pesos e os dados originais.

## Natureza do projeto

Projeto desenvolvido para fins acadêmicos, como atividade prática de implementação de redes neurais artificiais em Python.
