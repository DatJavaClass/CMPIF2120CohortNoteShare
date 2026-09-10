## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## kNN Classification general use calculator, stdlib only
## general fork of knn_classification_calc.py: any csv, any dimension, any labels
## reuses helpers from knn_regression_calc_GenUse.py (same folder)
## no flags = prompts, -h = flags

import sys
from collections import Counter
import knn_regression_calc_GenUse as kr
from knn_regression_calc_GenUse import hr, num, row, weights, rank, holdout, interactive, parser, setup, query_text

def classes(train): return list(dict.fromkeys(t[-1] for t in train)) ## first seen order

def vote(ranked, k, weighted): ## ranked = (distance, label) pairs
    c, w = Counter(), weights([d for d, _ in ranked[:k]]) if weighted else [1] * k
    for a, (_, l) in zip(w, ranked): c[l] += a
    top = max(c.values())
    winners = [l for l, n in c.items() if n == top]
    return c, winners[0] if len(winners) == 1 else None, winners

def predict(train, q, k, metric, std, weighted, verbose=True, sweep=0):
    cls = classes(train)
    if verbose and not weighted and k % len(cls) == 0: print(f"WARNING: k={k} is a multiple of {len(cls)} classes, ties possible")
    X, y, d = rank(train, q, metric, std, verbose)
    ranked = [(di, y[i]) for di, i in d]
    if verbose:
        hr(f"STEP 4: rank, keep the {k} nearest")
        print(row(["rank", "#", "distance"] + kr.FEATURES + [kr.TARGET, "kept"]))
        for r, (di, i) in enumerate(d, 1): print(row([r, i, num(di)] + list(X[i]) + [y[i], "yes" if r <= k else ""]))
        if k < len(d) and d[k - 1][0] == d[k][0]: print(f"NOTE: rank {k} and {k + 1} tie on distance, lower # kept")

    c, pred, winners = vote(ranked, k, weighted)
    if verbose:
        hr("STEP 5: " + ("weighted vote, weight = 1 / distance" if weighted else "majority vote"))
        for l in cls: print(f"{l}: {num(c[l])} / {num(sum(c.values()))}")
        print(f"PREDICTION: {pred}" if pred else f"TIE between {', '.join(winners)}, change k")

    if verbose and sweep:
        hr(f"SWEEP: decision for k = 1..{sweep}")
        print(row(["k"] + cls + ["decision"]))
        for kk in range(1, min(sweep, len(train)) + 1):
            ck, pk, _ = vote(ranked, kk, weighted)
            print(row([kk] + [num(ck[l]) for l in cls] + [pk or "tie"]))
    return pred

def evaluate(test_idx, a):
    cls = classes(kr.DATA)
    hr(f"EVAL: leave one out, {len(kr.DATA)} rows" if a.loo else f"EVAL: {len(kr.DATA) - len(test_idx)} train / {len(test_idx)} test (masked #{test_idx}, seed={a.seed})")
    pairs = [(i, kr.DATA[i][-1], predict(holdout(test_idx, i, a.loo), list(kr.DATA[i][:-1]), a.k, a.metric, a.std, a.weighted, verbose=False)) for i in test_idx]
    print(row(["#", "truth", "pred", "hit"]))
    for i, t, p in pairs: print(row([i, t, p or "tie", "yes" if t == p else "no"]))
    hits = sum(t == p for _, t, p in pairs)
    print(f"accuracy = {hits} / {len(pairs)} = {num(hits / len(pairs))}")

    hr("CONFUSION: rows = truth, cols = predicted")
    print(row(["truth\\pred"] + cls + ["tie"]))
    for t in cls: print(row([t] + [sum(1 for _, tt, p in pairs if tt == t and p == c) for c in cls] + [sum(1 for _, tt, p in pairs if tt == t and p is None)]))
    f1s = []
    for c in cls: ## one vs rest per class
        tp = sum(1 for _, t, p in pairs if t == c and p == c)
        fp, fn = sum(1 for _, t, p in pairs if t != c and p == c), sum(1 for _, t, p in pairs if t == c and p != c)
        prec, rec = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        print(f"{c}: TP={tp} FP={fp} FN={fn}  precision={num(prec)} recall={num(rec)} F1={num(f1s[-1])}")
    print(f"macro F1 = {num(sum(f1s) / len(f1s))}")

def main():
    ap = parser("kNN classification on any data set, step by step")
    ap.add_argument("--sweep", type=int, help="also show the decision for k = 1..N")
    a = interactive(ap, [("--sweep", "sweep decisions for k = 1..N (blank skips)")]) if len(sys.argv) == 1 else ap.parse_args()
    qs, idx = setup(a, str)
    a.brief or print(f"classes: {', '.join(classes(kr.DATA))}")
    for q in qs: print(f"\nPREDICTION for {query_text(q)} with k={a.k}: {predict(kr.DATA, q, a.k, a.metric, a.std, a.weighted, not a.brief, a.sweep or 0) or 'tie'}")
    if idx: evaluate(idx, a)

if __name__ == "__main__": main()
