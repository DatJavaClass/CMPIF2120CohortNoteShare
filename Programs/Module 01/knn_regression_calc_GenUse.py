## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## kNN Regression general use calculator, stdlib only
## general fork of knn_regression_calc.py: any csv, any dimension
## no flags = prompts, -h = flags

import argparse, math, os, random, re, sys

FEATURES, TARGET, DATA = [], "y", [] ## filled by load()

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=12): return "".join(num(c).rjust(w) for c in cells)
def lst(v): return ", ".join(num(x) for x in v)
def dbg(tag, v): print(f"[dbg] {tag}: {v}")

def isnum(s):
    try: float(s); return True
    except ValueError: return False

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

def parse_rows(s): return [parse_csv(r) for r in s.split(";")] ## "a,b;c,d"

def read_rows(spec): ## file path, - for stdin, or inline a,b;c,d
    try:
        if spec == "-": text = sys.stdin.read()
        elif os.path.isfile(spec): text = open(spec, encoding="utf-8-sig").read()
        else: text = spec.replace(";", "\n")
    except OSError as e: sys.exit(f"cannot read {spec}: {e}")
    rows = [re.split(r"\s*,\s*|\s+", l.strip()) for l in text.splitlines() if l.strip() and not l.lstrip().startswith("#")]
    if not rows: sys.exit("no data rows")
    if any(len(r) != len(rows[0]) for r in rows): sys.exit("ragged rows, every row needs the same column count")
    return rows

def header(rows): ## names + body, header row auto detected
    head = len(rows) > 1 and any(not isnum(a) and isnum(b) for a, b in zip(rows[0], rows[1]))
    return (rows[0] if head else [f"c{j}" for j in range(len(rows[0]))]), (rows[1:] if head else rows)

def col(names, key): ## column by name or index
    if key in names: return names.index(key)
    if key.isdigit() and int(key) < len(names): return int(key)
    sys.exit(f"no column {key}, have {', '.join(names)}")

def load(spec, target=None, feats=None, cast=float): ## last column = target by default
    global FEATURES, TARGET, DATA
    names, rows = header(read_rows(spec))
    t = col(names, target) if target else len(names) - 1
    f = [col(names, k) for k in feats.split(",")] if feats else [j for j in range(len(names)) if j != t]
    if t in f: sys.exit("target cannot also be a feature")
    try: DATA = [tuple(float(r[j]) for j in f) + (cast(r[t]),) for r in rows]
    except ValueError: sys.exit("non numeric cell, pick numeric columns with --features")
    FEATURES, TARGET = [names[j] for j in f], names[t]
    return DATA

def zstats(pts): ## mean and pop std per feature
    n, cols = len(pts), list(zip(*pts))
    mu = [sum(c) / n for c in cols]
    sd = [math.sqrt(sum((x - m) ** 2 for x in c) / n) for c, m in zip(cols, mu)]
    return mu, sd

def zscore(p, mu, sd): return [(x - m) / s if s else 0.0 for x, m, s in zip(p, mu, sd)]

def dist(a, b, metric):
    match metric:
        case "manhattan": return sum(abs(x - y) for x, y in zip(a, b))
        case "chebyshev": return max(abs(x - y) for x, y in zip(a, b))
        case _: return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def dist_text(a, b, metric): ## expanded formula for the printout
    terms = [f"|{num(x)}-{num(y)}|" for x, y in zip(a, b)]
    match metric:
        case "manhattan": return " + ".join(terms)
        case "chebyshev": return "max(" + ", ".join(terms) + ")"
        case _: return "sqrt(" + " + ".join(f"({num(x)}-{num(y)})^2" for x, y in zip(a, b)) + ")"

def weights(ds): return [float(d == 0) for d in ds] if 0 in ds else [1 / d for d in ds] ## exact hits take over

def rank(train, q, metric, std, verbose=True): ## standardize, distance, sort
    X, y = [t[:-1] for t in train], [t[-1] for t in train]
    if std:
        mu, sd = zstats(X)
        Xs, qs = [zscore(p, mu, sd) for p in X], zscore(q, mu, sd)
        if verbose:
            hr("STEP 2: standardize (z = (x - mean) / std, ddof=0)")
            print(row(["feature", "mean", "std"]))
            for f, m, s in zip(FEATURES, mu, sd): print(row([f, num(m), num(s)]))
            print(f"query ({lst(q)}) -> ({lst(qs)})")
    else: Xs, qs = X, q

    if verbose:
        hr(f"STEP 3: {metric} distance to query ({lst(q)})")
        print(row(["#", "distance"], 10) + "  formula")
    d = []
    for i, p in enumerate(Xs):
        di = dist(qs, p, metric)
        d.append((di, i))
        if verbose: print(row([i, num(di)], 10) + "  " + dist_text(qs, p, metric))
    d.sort() ## ties break on lower index
    return X, y, d

def predict(train, q, k, metric, std, weighted, verbose=True):
    X, y, d = rank(train, q, metric, std, verbose)
    nn = d[:k]
    if verbose:
        hr(f"STEP 4: {k} nearest neighbors")
        print(row(["rank", "#", "distance"] + FEATURES + [TARGET]))
        for r, (di, i) in enumerate(nn, 1): print(row([r, i, num(di)] + list(X[i]) + [y[i]]))
        if k < len(d) and d[k - 1][0] == d[k][0]: print(f"NOTE: rank {k} and {k + 1} tie on distance, lower # kept")

    vals, w = [y[i] for _, i in nn], weights([di for di, _ in nn]) if weighted else [1.0] * k
    pred = sum(a * v for a, v in zip(w, vals)) / sum(w)
    if verbose:
        hr("STEP 5: predict = sum(w * value) / sum(w), w = " + ("1 / distance" if weighted else "1 (plain mean)"))
        print(f"({' + '.join((f'{num(a)}*' if weighted else '') + num(v) for a, v in zip(w, vals))}) / {num(sum(w))} = {num(pred)}")
    return pred

def show_data(pts, title="STEP 1: data set"):
    hr(title)
    print(row(["#"] + FEATURES + [TARGET]))
    for i, p in enumerate(pts): print(row([i] + list(p)))

def split_idx(a): ## --test ids, --split fraction, or --loo
    if a.loo: return list(range(len(DATA)))
    if a.test: return sorted(set(parse_csv(a.test, int)))
    if a.split:
        random.seed(a.seed)
        return sorted(random.sample(range(len(DATA)), max(1, round(len(DATA) * a.split))))
    return None

def holdout(test_idx, i, loo): return [p for j, p in enumerate(DATA) if j != i and (loo or j not in test_idx)] ## training rows for test row i

def evaluate(test_idx, a):
    hr(f"EVAL: leave one out, {len(DATA)} rows" if a.loo else f"EVAL: {len(DATA) - len(test_idx)} train / {len(test_idx)} test (masked #{test_idx}, seed={a.seed})")
    pairs = [(i, DATA[i][-1], predict(holdout(test_idx, i, a.loo), list(DATA[i][:-1]), a.k, a.metric, a.std, a.weighted, verbose=False)) for i in test_idx]
    print(row(["#", "truth p", "pred q", "|p-q|", "(p-q)^2"]))
    for i, p, q in pairs: print(row([i, num(p), num(q), num(abs(p - q)), num((p - q) ** 2)]))
    n, pm = len(pairs), sum(p for _, p, _ in pairs) / len(pairs)
    sae, sse, sst = sum(abs(p - q) for _, p, q in pairs), sum((p - q) ** 2 for _, p, q in pairs), sum((p - pm) ** 2 for _, p, _ in pairs)
    hr("MAE = sum|p-q| / n")
    print(f"{num(sae)} / {n} = {num(sae / n)}")
    hr("RMSE = sqrt(sum(p-q)^2 / n)")
    print(f"sqrt({num(sse)} / {n}) = sqrt({num(sse / n)}) = {num(math.sqrt(sse / n))}")
    hr("R^2 = 1 - SSE / SST, SST = sum(p - mean p)^2")
    print(f"1 - {num(sse)} / {num(sst)} = {num(1 - sse / sst)}" if sst else "SST = 0, R^2 undefined")

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def interactive(ap, extra=()): ## extra = sibling prompts, (flag, question) pairs
    argv = ["-d", ask("data: csv path, - for stdin, or rows a,b,y;c,d,y")]
    for flag, q in (("--target", "target column name or index (blank = last)"), ("--features", "feature columns (blank = all others)"), ("-q", "query points x,y;x,y (blank skips)")) + tuple(extra):
        v = ask(q)
        if v: argv += [flag, v]
    argv += ["-k", ask("k", "3")]
    if ask("standardize features? y/N", "n").lower() == "y": argv.append("--std")
    if ask("weight neighbors by 1/distance? y/N", "n").lower() == "y": argv.append("--weighted")
    argv += ["--metric", ask("metric euclidean/manhattan/chebyshev", "euclidean")]
    e = ask("evaluate: test ids 1,4 or a fraction 0.2 or loo (blank skips)")
    if e == "loo": argv.append("--loo")
    elif e: argv += ["--split" if "." in e else "--test", e]
    return ap.parse_args(argv)

def parser(desc): ## flags shared with the classification sibling
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("-d", "--data", help="csv file, - for stdin, or inline rows a,b,y;c,d,y (header optional)")
    ap.add_argument("--target", help="target column name or index, default last")
    ap.add_argument("--features", help="feature columns, default every other column")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("-q", "--query", help="query points x,y;x,y")
    ap.add_argument("--std", action="store_true", help="z-score features before distance")
    ap.add_argument("--weighted", action="store_true", help="weight neighbors by 1 / distance")
    ap.add_argument("--metric", choices=["euclidean", "manhattan", "chebyshev"], default="euclidean")
    ap.add_argument("--test", help="row indices to mask, e.g. 1,4,7")
    ap.add_argument("--split", type=float, help="random test fraction, e.g. 0.2")
    ap.add_argument("--loo", action="store_true", help="leave one out over every row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--brief", action="store_true", help="skip the step by step printout")
    return ap

def setup(a, cast=float): ## load, validate, return queries and test ids
    if not a.data: sys.exit("need -d data")
    if a.split is not None and not 0 < a.split < 1: sys.exit("split must be between 0 and 1")
    load(a.data, a.target, a.features, cast)
    if not 0 < a.k <= len(DATA): sys.exit(f"k must be 1..{len(DATA)}")
    qs, idx = parse_rows(a.query) if a.query else [], split_idx(a)
    if any(len(q) != len(FEATURES) for q in qs): sys.exit(f"query needs {len(FEATURES)} values: {', '.join(FEATURES)}")
    if idx:
        if any(i < 0 or i >= len(DATA) for i in idx): sys.exit("test index out of range")
        if len(DATA) - (1 if a.loo else len(idx)) < a.k: sys.exit("not enough training rows for k")
    if not qs and not idx: sys.exit("nothing to do: give -q and/or --test, --split, --loo")
    a.brief or show_data(DATA)
    return qs, idx

def query_text(q): return ", ".join(f"{f}={num(v)}" for f, v in zip(FEATURES, q))

def main():
    ap = parser("kNN regression on any data set, step by step")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    qs, idx = setup(a)
    for q in qs: print(f"\nPREDICTION for {query_text(q)} with k={a.k}: {num(predict(DATA, q, a.k, a.metric, a.std, a.weighted, not a.brief))}")
    if idx: evaluate(idx, a)

if __name__ == "__main__": main()
