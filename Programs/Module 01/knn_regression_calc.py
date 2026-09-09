## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## kNN Regression bare metal calculator, stdlib only
## no flags = prompts, -h = flags

import argparse, math, random, sys

## edit the data set here, values = target
FEATURES = ["square_feet", "year_built"]
TARGET = "price"
DATA = [
    (1500, 1998, 350), (2100, 2005, 480), (1650, 1980, 320), \
    (2500, 2010, 610), (1850, 1995, 400), (1400, 1987, 290), \
    (2200, 2015, 650), (1700, 2000, 370), (1950, 1992, 410), \
    (2300, 2008, 590)]

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=12): return "".join(num(c).rjust(w) for c in cells)
def dbg(tag, v): print(f"[dbg] {tag}: {v}")

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

def zstats(pts): ## mean and pop std per feature
    n, cols = len(pts), list(zip(*pts))
    mu = [sum(c) / n for c in cols]
    sd = [math.sqrt(sum((x - m) ** 2 for x in c) / n) for c, m in zip(cols, mu)]
    return mu, sd

def zscore(p, mu, sd): return [(x - m) / s if s else 0.0 for x, m, s in zip(p, mu, sd)]

def dist(a, b, metric):
    if metric == "manhattan": return sum(abs(x - y) for x, y in zip(a, b))
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def dist_text(a, b, metric): ## expanded formula for the printout
    if metric == "manhattan": return " + ".join(f"|{num(x)}-{num(y)}|" for x, y in zip(a, b))
    return "sqrt(" + " + ".join(f"({num(x)}-{num(y)})^2" for x, y in zip(a, b)) + ")"

def predict(train, q, k, metric, std, verbose=True):
    X, y = [t[:-1] for t in train], [t[-1] for t in train]

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
    nn = d[:k]
    if verbose:
        hr(f"STEP 4: {k} nearest neighbors")
        print(row(["rank", "#", "distance"] + FEATURES + [TARGET]))
        for r, (di, i) in enumerate(nn, 1): print(row([r, i, num(di)] + list(X[i]) + [y[i]]))

    vals = [y[i] for _, i in nn]
    pred = sum(vals) / k
    if verbose:
        hr("STEP 5: predict = mean of neighbor values")
        print(f"({' + '.join(num(v) for v in vals)}) / {k} = {num(pred)}")
    return pred

def show_data(pts, title="STEP 1: data set"):
    hr(title)
    print(row(["#"] + FEATURES + [TARGET]))
    for i, p in enumerate(pts): print(row([i] + list(p)))

def evaluate(test_idx, k, metric, std, seed):
    train = [p for i, p in enumerate(DATA) if i not in test_idx]
    test = [DATA[i] for i in test_idx]
    hr(f"EVAL: {len(train)} train / {len(test)} test (masked #{test_idx}, seed={seed})")
    pairs = []
    for i, t in zip(test_idx, test): ## p = truth, q = prediction
        p, q = t[-1], predict(train, list(t[:-1]), k, metric, std, verbose=False)
        pairs.append((i, p, q))

    print(row(["#", "truth p", "pred q", "|p-q|", "(p-q)^2"]))
    for i, p, q in pairs: print(row([i, num(p), num(q), num(abs(p - q)), num((p - q) ** 2)]))
    n = len(pairs)
    sae, sse = sum(abs(p - q) for _, p, q in pairs), sum((p - q) ** 2 for _, p, q in pairs)
    hr("MAE = sum|p-q| / n")
    print(f"{num(sae)} / {n} = {num(sae / n)}")
    hr("RMSE = sqrt(sum(p-q)^2 / n)")
    print(f"sqrt({num(sse)} / {n}) = sqrt({num(sse / n)}) = {num(math.sqrt(sse / n))}")

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def ask_data():
    global FEATURES, TARGET, DATA
    FEATURES = ask("feature names, comma separated", ",".join(FEATURES)).split(",")
    TARGET = ask("target name", TARGET)
    print(f"rows as {','.join(FEATURES)},{TARGET}; blank line ends")
    rows = []
    while True:
        line = input(f"row {len(rows)}: ").strip()
        if not line: break
        try: rows.append(tuple(float(x) for x in line.split(",")))
        except ValueError: print("  not numeric, retry"); continue
        if len(rows[-1]) != len(FEATURES) + 1: rows.pop(); print(f"  need {len(FEATURES) + 1} values, retry")
    if rows: DATA = rows
    else: print("no rows, keeping built in data")

def interactive(ap):
    if ask("edit the data set? y/N", "n").lower() == "y": ask_data()
    argv = ["-k", ask("k", "4"), "-q", ask("query " + ",".join(FEATURES), "2000,1999")]
    if ask("standardize features? y/N", "n").lower() == "y": argv.append("--std")
    argv += ["--metric", ask("metric euclidean/manhattan", "euclidean")]
    t = ask("test rows to mask for MAE/RMSE, e.g. 1,4,7 (blank skips)")
    if t: argv += ["--test", t]
    return ap.parse_args(argv)

def main():
    ap = argparse.ArgumentParser(description="kNN regression, step by step")
    ap.add_argument("-k", type=int, default=4)
    ap.add_argument("-q", "--query", default="2000,1999", help="feature values, comma separated")
    ap.add_argument("--std", action="store_true", help="z-score features before distance")
    ap.add_argument("--metric", choices=["euclidean", "manhattan"], default="euclidean")
    ap.add_argument("--test", help="row indices to mask for MAE/RMSE, e.g. 1,4,7")
    ap.add_argument("--split", type=float, help="random test fraction, e.g. 0.2")
    ap.add_argument("--seed", type=int, default=0)
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()

    q = parse_csv(a.query)
    if len(q) != len(FEATURES): sys.exit(f"query needs {len(FEATURES)} values: {FEATURES}")
    if not 0 < a.k <= len(DATA): sys.exit(f"k must be 1..{len(DATA)}")
    show_data(DATA)
    pred = predict(DATA, q, a.k, a.metric, a.std)
    print(f"\nPREDICTION for {dict(zip(FEATURES, q))} with k={a.k}: {num(pred)}")

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
