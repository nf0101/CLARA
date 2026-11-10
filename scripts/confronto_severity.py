"""Confronto tra la mappatura argmax attuale e la formula continua.

Uso:  python3 confronto_severity.py <dir con i CSV>
Gira il Transformer una sola volta per lamentela; le due severity
sono derivate offline dalle stesse probabilita'.
"""
import csv
import sys
from pathlib import Path

from transformers import pipeline

sys.path.insert(0, "..")
from src.nlp import sentiment_module as M  # noqa: E402

DATA = Path(sys.argv[1] if len(sys.argv) > 1 else ".")

# Quote D40 per client in RDD2022 (train) -> vettore Oracle di riferimento
D40 = {"czech": 197, "india": 3187, "japan": 2243,
       "norway": 461, "united_states": 135, "china_motorbike": 235}
_tot = sum(D40.values())
ORACLE = {k: v / _tot for k, v in D40.items()}

pipe = pipeline("sentiment-analysis", model=M._MODEL_NAME,
                tokenizer=M._MODEL_NAME, device=-1, truncation=True,
                max_length=512, top_k=None)


def probs(text):
    return {d["label"].lower(): float(d["score"]) for d in pipe(text)[0]}


def st_attuale(p):
    lab = max(p, key=p.get)
    if lab == "negative":
        return 0.5 + 0.5 * p[lab]
    if lab == "positive":
        return 0.5 - 0.5 * p[lab]
    return 0.5


def st_alternativa(p):
    return p["negative"] + 0.5 * p["neutral"]


def finale(st, text):
    kh, kl, km = M._keyword_boosts(text)
    prelim = (M._TRANSFORMER_WEIGHT * st + (1 - M._TRANSFORMER_WEIGHT) * 0.5
              + kh + kl + M._caps_boost(text) + M._exclamation_boost(text))
    return max(0.0, min(1.0, prelim + km * (0.5 - prelim)))


def urgenza(rows, sev):
    u = {}
    for r, s in zip(rows, sev):
        u[r["expected_country"]] = u.get(r["expected_country"], 0.0) + s
    u = {k: max(v, 0.01) for k, v in u.items()}
    tot = sum(u.values())
    return {k: v / tot for k, v in u.items()}


def l1_da_oracle(u):
    return sum(abs(u.get(k, 0.0) - ORACLE[k]) for k in ORACLE)


for nome in ["synthetic_complaints.csv", "synthetic_complaints_realistic.csv"]:
    path = DATA / nome
    if not path.exists():
        print(f"[salto] {path} non trovato")
        continue
    rows = list(csv.DictReader(open(path)))
    P = [probs(r["text"]) for r in rows]

    sev_a = [finale(st_attuale(p), r["text"]) for p, r in zip(P, rows)]
    sev_b = [finale(st_alternativa(p), r["text"]) for p, r in zip(P, rows)]

    cat_a = [M.severity_to_category(s) for s in sev_a]
    cat_b = [M.severity_to_category(s) for s in sev_b]
    atteso = [r["expected_severity"] for r in rows]

    acc_a = sum(x == y for x, y in zip(cat_a, atteso)) / len(rows)
    acc_b = sum(x == y for x, y in zip(cat_b, atteso)) / len(rows)
    cambiate = sum(x != y for x, y in zip(cat_a, cat_b))
    scarto = sum(abs(x - y) for x, y in zip(sev_a, sev_b)) / len(rows)

    print(f"\n=== {nome} ({len(rows)} lamentele) ===")
    print(f"accuracy attuale     : {acc_a:.3f}")
    print(f"accuracy alternativa : {acc_b:.3f}")
    print(f"categorie cambiate   : {cambiate}")
    print(f"scarto medio severity: {scarto:.4f}")

    ua, ub = urgenza(rows, sev_a), urgenza(rows, sev_b)
    print("\nclient            attuale  alternativa  oracle")
    for k in sorted(ORACLE):
        print(f"{k:18s} {ua.get(k,0):.4f}     {ub.get(k,0):.4f}   {ORACLE[k]:.4f}")
    print(f"\ndistanza L1 da Oracle: attuale {l1_da_oracle(ua):.4f}, "
          f"alternativa {l1_da_oracle(ub):.4f}")