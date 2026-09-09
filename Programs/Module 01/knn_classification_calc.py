## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## kNN Classification bare metal calculator, stdlib only
## reuses helpers from knn_regression_calc.py (same folder)
## no flags = prompts, -h = flags

import argparse, random, sys
from collections import Counter
from knn_regression_calc import hr, num, row, ask, parse_csv, zstats, zscore, dist, dist_text

## edit the data set here, last column = class label
FEATURES = ["x", "y"]
TARGET = "label"
DATA = [(1, 1, "A"), (2, 3, "B"), (3, 2, "B"), (6, 5, "A"), (7, 8, "A")]

def classes(train): return list(dict.fromkeys(t[-1] for t in train)) ## first seen order

def vote(labels, k):
    c = Counter(labels[:k])
    top = max(c.values())
    winners = [l for l, n in c.items() if n == top]
    return c, winners[0] if len(winners) == 1 else None, winners

def predict(train, q, k, metric, std, verbose=True, sweep=0):
    X, y, cls = [t[:-1] for t in train], [t[-1] for t in train], classes(train)
    if verbose and k % len(cls) == 0: print(f"WARNING: k={k} is a multiple of {len(cls)} classes, ties possible")

    if std:
        mu, sd = zstats(X)
        Xs, qs = [zscore(p, mu, sd) for p in X], zscore(q, mu, sd)
        if verbose:
            hr("STEP 2: standardize (z = (x - mean) / std, ddof=0)")
            print(row(["feature", "mean", "std"]))
            for f, m, s in zip(FEATURES, mu, sd): print(row([f, num(m), num(s)]))
            print(f"query {q} -> {[num(v) for v in qs]}")
    else: Xs, qs = X, q

    if verbose:
        hr(f"STEP 3: {metric} distance to query {q}")
        print(row(["#", "distance"], 10) + "  formula")
    d = []
    for i, p in enumerate(Xs):
        di = dist(qs, p, metric)
        d.append((di, i))
        if verbose: print(row([i, num(di)], 10) + "  " + dist_text(qs, p, metric))

    d.sort() ## ties break on lower index
    ranked = [y[i] for _, i in d]
    if verbose:
        hr(f"STEP 4: rank, keep the {k} nearest")
        print(row(["rank", "#", "distance"] + FEATURES + [TARGET, "kept"]))
        for r, (di, i) in enumerate(d, 1): print(row([r, i, num(di)] + list(X[i]) + [y[i], "yes" if r <= k else ""]))
        if k < len(d) and d[k - 1][0] == d[k][0]: print(f"NOTE: rank {k} and {k + 1} tie on distance, lower # kept")

    c, pred, winners = vote(ranked, k)
    if verbose:
        hr("STEP 5: majority vote")
        for l in cls: print(f"{l}: {c[l]} / {k}")
        print(f"PREDICTION: {pred}" if pred else f"TIE between {', '.join(winners)}, change k")

    if verbose and sweep:
        hr(f"SWEEP: decision for k = 1..{sweep}")
        print(row(["k"] + cls + ["decision"]))
        for kk in range(1, min(sweep, len(train)) + 1):
            ck, pk, wk = vote(ranked, kk)
            print(row([kk] + [ck[l] for l in cls] + [pk or "tie"]))
    return pred

def evaluate(test_idx, k, metric, std, seed):
    train = [p for i, p in enumerate(DATA) if i not in test_idx]
    test, cls = [DATA[i] for i in test_idx], classes(DATA)
    hr(f"EVAL: {len(train)} train / {len(test)} test (masked #{test_idx}, seed={seed})")
    pairs = [(i, t[-1], predict(train, list(t[:-1]), k, metric, std, verbose=False)) for i, t in zip(test_idx, test)]
    print(row(["#", "truth", "pred", "hit"]))
    for i, t, p in pairs: print(row([i, t, p or "tie", "yes" if t == p else "no"]))
    hits = sum(t == p for _, t, p in pairs)
    print(f"accuracy = {hits} / {len(pairs)} = {num(hits / len(pairs))}")

    hr("CONFUSION: rows = truth, cols = predicted")
    print(row(["truth\\pred"] + cls + ["tie"]))
    for t in cls: print(row([t] + [sum(1 for _, tt, p in pairs if tt == t and p == c) for c in cls] + [sum(1 for _, tt, p in pairs if tt == t and p is None)]))
    for c in cls: ## one vs rest per class
        tp = sum(1 for _, t, p in pairs if t == c and p == c)
        fp, fn = sum(1 for _, t, p in pairs if t != c and p == c), sum(1 for _, t, p in pairs if t == c and p != c)
        prec, rec = tp / (tp + fp) if tp + fp else 0.0, tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{c}: TP={tp} FP={fp} FN={fn}  precision={num(prec)} recall={num(rec)} F1={num(f1)}")

def ask_data():
    global FEATURES, TARGET, DATA
    FEATURES = ask("feature names, comma separated", ",".join(FEATURES)).split(",")
    TARGET = ask("label name", TARGET)
    print(f"rows as {','.join(FEATURES)},{TARGET}; blank line ends")
    rows = []
    while True:
        line = input(f"row {len(rows)}: ").strip()
        if not line: break
        parts = line.split(",")
        if len(parts) != len(FEATURES) + 1: print(f"  need {len(FEATURES) + 1} values, retry"); continue
        try: rows.append(tuple(float(x) for x in parts[:-1]) + (parts[-1].strip(),))
        except ValueError: print("  features not numeric, retry")
    if rows: DATA = rows
    else: print("no rows, keeping built in data")

def interactive(ap):
    if ask("edit the data set? y/N", "n").lower() == "y": ask_data()
    argv = ["-k", ask("k", "3"), "-q", ask("query " + ",".join(FEATURES), "3,3")]
    if ask("standardize features? y/N", "n").lower() == "y": argv.append("--std")
    argv += ["--metric", ask("metric euclidean/manhattan", "euclidean")]
    s = ask("sweep decisions for k = 1..N (blank skips)")
    if s: argv += ["--sweep", s]
    t = ask("test rows to mask for accuracy, e.g. 1,4 (blank skips)")
    if t: argv += ["--test", t]
    return ap.parse_args(argv)

def main():
    ap = argparse.ArgumentParser(description="kNN classification, step by step")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("-q", "--query", default="3,3", help="feature values, comma separated")
    ap.add_argument("--std", action="store_true", help="z-score features before distance")
    ap.add_argument("--metric", choices=["euclidean", "manhattan"], default="euclidean")
    ap.add_argument("--sweep", type=int, help="also show the decision for k = 1..N")
    ap.add_argument("--test", help="row indices to mask for accuracy, e.g. 1,4")
    ap.add_argument("--split", type=float, help="random test fraction, e.g. 0.2")
    ap.add_argument("--seed", type=int, default=0)
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()

    q = parse_csv(a.query)
    if len(q) != len(FEATURES): sys.exit(f"query needs {len(FEATURES)} values: {FEATURES}")
    if not 0 < a.k <= len(DATA): sys.exit(f"k must be 1..{len(DATA)}")
    hr("STEP 1: data set")
    print(row(["#"] + FEATURES + [TARGET]))
    for i, p in enumerate(DATA): print(row([i] + list(p)))
    print(f"classes: {', '.join(classes(DATA))}")
    predict(DATA, q, a.k, a.metric, a.std, sweep=a.sweep or 0)

    test_idx = None
    if a.test: test_idx = sorted(set(parse_csv(a.test, int)))
    elif a.split:
        random.seed(a.seed)
        test_idx = sorted(random.sample(range(len(DATA)), max(1, round(len(DATA) * a.split))))
    if test_idx:
        if any(i < 0 or i >= len(DATA) for i in test_idx): sys.exit("test index out of range")
        if len(DATA) - len(test_idx) < a.k: sys.exit("not enough training rows for k")
        evaluate(test_idx, a.k, a.metric, a.std, a.seed)

if __name__ == "__main__": main()
