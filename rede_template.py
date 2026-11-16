"""
Teste: implementação de uma rede neural já treinada (propagação direta)
Matemática Computacional - Prof. Dr. Alexsandro O. A.

Nome(s): Maria Taliciane Pereira da Silva

Regras:
- Use apenas Python, NumPy e pandas.
- É proibido usar PyTorch, TensorFlow, Keras, sklearn ou qualquer rede neural pronta.
- As funções implementam a propagação direta descrita no README.
- Rode `python rede_template.py`: o teste de validação mostra OK ou FALHOU.
"""

import json
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Leitura dos pesos (pronta)
# ---------------------------------------------------------------------------

def carregar_rede(arquivo="pesos_rede.json"):
    """Lê o JSON e devolve um dicionário com os pesos como arrays NumPy."""
    with open(arquivo, encoding="utf-8") as f:
        cfg = json.load(f)
    for chave in ["W1", "b1", "W2", "b2", "W3", "b3", "normalizacao_divisor"]:
        cfg[chave] = np.array(cfg[chave], dtype=float)
    return cfg


# ---------------------------------------------------------------------------
# Funções da rede
# ---------------------------------------------------------------------------

def normalizar(entrega_bruta, divisores):
    """Divida cada atributo bruto pelo divisor correspondente.

    entrega_bruta = [distancia_km, peso_kg, urgencia, valor_declarado]
    Retorne o vetor x (tamanho 4) que entra na rede.
    """
    return np.asarray(entrega_bruta, dtype=float) / np.asarray(divisores, dtype=float)


def relu(z):
    """ReLU(z) = max(0, z), elemento a elemento."""
    return np.maximum(0, np.asarray(z, dtype=float))


def tanh(z):
    """Tangente hiperbólica, elemento a elemento."""
    return np.tanh(np.asarray(z, dtype=float))


def softmax(z):
    """Calcule a softmax de z com estabilização numérica.

    Subtrair max(z) não muda o resultado e evita overflow.
    """
    z = np.asarray(z, dtype=float)
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)


def camada(W, b, x, ativacao):
    """Devolva (z, h), com z = W @ x + b e h = ativacao(z)."""
    z = np.asarray(W, dtype=float) @ np.asarray(x, dtype=float) + np.asarray(b, dtype=float)
    return z, ativacao(z)


def propagar(rede, x):
    """Faça a propagação direta pelas 3 camadas de pesos.

    Retorne um dicionário com as chaves
    "z1", "h1", "z2", "h2", "z3", "p"  (p = probabilidades da saída).
    Ativações: camada 1 -> ReLU, camada 2 -> tanh, camada 3 -> softmax.
    """
    z1, h1 = camada(rede["W1"], rede["b1"], x, relu)
    z2, h2 = camada(rede["W2"], rede["b2"], h1, tanh)
    z3 = rede["W3"] @ h2 + rede["b3"]
    p = softmax(z3)
    return {"z1": z1, "h1": h1, "z2": z2, "h2": h2, "z3": z3, "p": p}


def classificar(rede, entrega_bruta):
    """Normalize atributos brutos e retorne a classe mais provável."""
    x = normalizar(entrega_bruta, rede["normalizacao_divisor"])
    resultado = propagar(rede, x)
    indice = int(np.argmax(resultado["p"]))
    return rede["classes"][indice], float(resultado["p"][indice])


# ---------------------------------------------------------------------------
# Teste de validação (exemplo resolvido no enunciado)
# ---------------------------------------------------------------------------

def _confere(nome, obtido, esperado, tol=1e-3):
    ok = np.allclose(np.asarray(obtido, float), np.asarray(esperado, float), atol=tol)
    print(f"[{'OK' if ok else 'FALHOU'}] {nome}")
    if not ok:
        print("    esperado:", np.round(esperado, 4), "\n    obtido:  ", np.round(obtido, 4))


def validar(rede):
    print("=" * 60 + "\nVALIDAÇÃO: x = (0.60, 0.40, 0.50, 0.20)\n" + "=" * 60)
    x = np.array([0.60, 0.40, 0.50, 0.20])
    r = propagar(rede, x)
    _confere("z1", r["z1"], [1.602, -0.417, 1.433, -1.171, -0.521, 1.274])
    _confere("h1", r["h1"], [1.602, 0.000, 1.433, 0.000, 0.000, 1.274])
    _confere("z2", r["z2"], [2.5760, -0.9262, 4.1550, 0.0951])
    _confere("h2", r["h2"], [0.9885, -0.7288, 0.9995, 0.0948])
    _confere("z3", r["z3"], [-3.5755, 4.4592, -1.0141])
    _confere("p", r["p"], [0.0003, 0.9955, 0.0042])
    entrega_bruta = x * rede["normalizacao_divisor"]
    classe, prob = classificar(rede, entrega_bruta)
    print(f"Classe prevista: {classe} ({100 * prob:.2f}%)  -> esperado: SEDEX (99.55%)")


def resolver_trabalho(rede, entregas):
    """Mostre as contas e o gabarito das dez perguntas do teste de conhecimento."""
    fmt = lambda valores, casas: np.array2string(
        np.asarray(valores), precision=casas, suppress_small=True
    )

    print("\n" + "=" * 60 + "\nRESOLUÇÃO DO TRABALHO\n" + "=" * 60)

    pesos = sum(rede[k].size for k in ["W1", "W2", "W3"])
    vieses = sum(rede[k].size for k in ["b1", "b2", "b3"])
    parametros = pesos + vieses
    print(
        f"\nP1 | Pesos: (6×4) + (4×6) + (3×4) = {pesos}"
        f"\n    Vieses: 6 + 4 + 3 = {vieses}"
        f"\n    Total: {pesos} + {vieses} = {parametros}"
    )

    x = np.array([0.50, 0.70, 0.40, 0.30])
    r = propagar(rede, x)
    zeros_relu = int(np.count_nonzero(r["h1"] == 0))
    indice = int(np.argmax(r["p"]))
    classe = rede["classes"][indice]

    print(f"\nEntrada normalizada das P2–P7: x = {fmt(x, 2)}")
    print("\nP2 | Aplicar a primeira camada e a ReLU:")
    print(f"    z1 = W1 @ x + b1 = {fmt(r['z1'], 3)}")
    print(f"    h1 = max(0, z1) = {fmt(r['h1'], 3)}")
    print(f"    Contagem de zeros em h1 = {zeros_relu}")
    print("\nP3 | Vetor h1 arredondado a 3 casas:")
    print(f"    h1 = {fmt(r['h1'], 3)}")
    print("\nP4 | Aplicar a segunda camada:")
    print(f"    z2 = W2 @ h1 + b2 = {fmt(r['z2'], 4)}")
    print("\nP5 | Aplicar tanh elemento a elemento:")
    print(f"    h2 = tanh(z2) = {fmt(r['h2'], 4)}")
    print("\nP6 | Calcular os logits de saída:")
    print(f"    z3 = W3 @ h2 + b3 = {fmt(r['z3'], 3)}")
    print("\nP7 | Softmax dos logits:")
    print(f"    p = softmax(z3) = {fmt(r['p'], 6)}")
    print(f"    Maior probabilidade: {classe} = {100 * r['p'][indice]:.2f}%")

    bruta8 = [150, 3, 0.95, 1200]
    x8 = normalizar(bruta8, rede["normalizacao_divisor"])
    r8 = propagar(rede, x8)
    classe8, prob8 = classificar(rede, bruta8)
    indice8 = int(np.argmax(r8["p"]))
    print("\nP8 | Classificar os dados brutos:")
    print(f"    x = {bruta8} / {fmt(rede['normalizacao_divisor'], 0)} = {fmt(x8, 2)}")
    print(f"    z1 = W1 @ x + b1 = {fmt(r8['z1'], 4)}")
    print(f"    h1 = ReLU(z1) = {fmt(r8['h1'], 4)}")
    print(f"    z2 = W2 @ h1 + b2 = {fmt(r8['z2'], 4)}")
    print(f"    h2 = tanh(z2) = {fmt(r8['h2'], 4)}")
    print(f"    z3 = W3 @ h2 + b3 = {fmt(r8['z3'], 4)}")
    print(f"    p = softmax(z3) = {fmt(r8['p'], 6)}")
    print(f"    Classe: {classe8} (índice {indice8}); {100 * prob8:.2f}%")

    resultados = []
    for _, linha in entregas.iterrows():
        bruta = [
            linha.distancia_km,
            linha.peso_kg,
            linha.urgencia,
            linha.valor_declarado,
        ]
        x_entrega = normalizar(bruta, rede["normalizacao_divisor"])
        r_entrega = propagar(rede, x_entrega)
        prevista, prob = classificar(rede, bruta)
        correta = prevista == linha.modalidade_real
        resultados.append(
            (int(linha.id_entrega), prevista, linha.modalidade_real, prob, correta)
        )
        print(
            f"\n    Entrega {int(linha.id_entrega)} | P9"
            f"\n      bruta / divisores = {fmt(x_entrega, 4)}"
            f"\n      logits = {fmt(r_entrega['z3'], 4)}"
            f"\n      probabilidades (PAC, SEDEX, SEDEX 10) = "
            f"{fmt(r_entrega['p'], 6)}"
            f"\n      prevista = {prevista} ({100 * prob:.2f}%), "
            f"real = {linha.modalidade_real} -> "
            f"{'acerto' if correta else 'erro'}"
        )

    corretas = sum(correta for _, _, _, _, correta in resultados)
    acuracia = 100 * corretas / len(resultados)
    erros = [
        (id_entrega, prevista, real)
        for id_entrega, prevista, real, _, correta in resultados
        if not correta
    ]
    print(
        f"\nP9 | Acurácia = acertos / total × 100"
        f" = {corretas} / {len(resultados)} × 100 = {acuracia:.0f}%"
    )
    print("\nP10 | Comparar classe prevista com a classe real:")
    print(f"     Erros (entrega, prevista, real): {erros}")

    print("\nGABARITO")
    print("P1: C  (73 parâmetros)")
    print(f"P2: A  ({zeros_relu} ativações iguais a zero)")
    print("P3: E  (h1 acima)")
    print("P4: B  (z2 acima)")
    print("P5: D  (h2 acima)")
    print("P6: C  (z3 acima)")
    print(f"P7: A  ({classe}, {100 * r['p'][indice]:.2f}%)")
    print(f"P8: E  ({classe8}, {100 * prob8:.2f}%)")
    print(f"P9: B  ({acuracia:.0f}%)")
    print("P10: D (entregas 5 e 6 previstas como PAC; real SEDEX)")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rede = carregar_rede()
    validar(rede)

    np.set_printoptions(precision=4, suppress=True)
    df = pd.read_csv("entregas_teste_rede.csv")
    resolver_trabalho(rede, df)
