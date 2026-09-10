## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## PCA general use calculator, stdlib only
## general fork of pca_calc.py: any csv, any dimension, or a covariance matrix
## no flags = prompts, -h = flags

import argparse, math, os, re, sys

FEATURES, DATA, IDS = [], [], None ## filled by load()

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=12): return "".join(num(c).rjust(w) for c in cells)
def table(head, rows, w=12): print(row(head, w)); [print(row(r, w)) for r in rows]
def lst(v): return "[" + ", ".join(num(x) for x in v) + "]"
def dbg(tag, v): print(f"[dbg] {tag}: {v}")
def pid(i): return IDS[i] if IDS else i ## display id

def isnum(s):
    try: float(s); return True
    except ValueError: return False

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

def parse_matrix(s): return [parse_csv(r) for r in s.split(";")] ## "a,b;c,d"

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

def load(spec, feats=None, idcol=None): ## numeric columns only, header auto
    global FEATURES, DATA, IDS
    names, rows = header(read_rows(spec))
    idj = col(names, idcol) if idcol else -1
    f = [col(names, k) for k in feats.split(",")] if feats else [j for j in range(len(names)) if j != idj and all(isnum(r[j]) for r in rows)]
    if not f or idj in f: sys.exit("need at least one numeric feature column, id column excluded")
    skip = [names[j] for j in range(len(names)) if j not in f and j != idj]
    if skip: print(f"unused column(s): {', '.join(skip)}")
    try: DATA = [tuple(float(r[j]) for j in f) for r in rows]
    except ValueError: sys.exit("non numeric cell in a feature column, pick columns with --features")
    FEATURES, IDS = [names[j] for j in f], [r[idj] for r in rows] if idj >= 0 else None
    return DATA

## vector helpers
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def unit(a): n = norm(a); return [x / n for x in a] if n else a
def matvec(C, v): return [dot(r, v) for r in C]

## column stats
def mean(c): return sum(c) / len(c)
def std(c, ddof=0): m = mean(c); return math.sqrt(sum((x - m) ** 2 for x in c) / (len(c) - ddof))
def pctl(c, p): ## linear interpolation, numpy default
    s, k = sorted(c), p * (len(c) - 1)
    f, w = int(k), k - int(k)
    return s[f] + (s[min(f + 1, len(s) - 1)] - s[f]) * w

def scale(cols, how, ddof):
    if how == "none": return cols, [("none", 0, 1)] * len(cols)
    out, stats = [], []
    for c in cols:
        match how: ## one variable vs constants = switch
            case "center": a, b = mean(c), 1
            case "zscore": a, b = mean(c), std(c, ddof)
            case "minmax": a, b = min(c), max(c) - min(c)
            case _: a, b = pctl(c, .25), pctl(c, .75) - pctl(c, .25) ## robust
        out.append([(x - a) / b if b else 0.0 for x in c])
        stats.append((how, a, b))
    return out, stats

def cov_matrix(cols, ddof):
    n, mu, p = len(cols[0]), [mean(c) for c in cols], len(cols)
    return [[sum((x - mu[i]) * (y - mu[j]) for x, y in zip(cols[i], cols[j])) / (n - ddof) for j in range(p)] for i in range(p)]

def eig_sym(C, sweeps=100, tol=1e-12): ## cyclic Jacobi, symmetric only
    p = len(C)
    A, V = [r[:] for r in C], [[float(i == j) for j in range(p)] for i in range(p)]
    for _ in range(sweeps):
        if sum(A[i][j] ** 2 for i in range(p) for j in range(p) if i != j) < tol: break
        for i in range(p):
            for j in range(i + 1, p):
                if abs(A[i][j]) < 1e-300: continue
                th = 0.5 * math.atan2(2 * A[i][j], A[j][j] - A[i][i])
                c, s = math.cos(th), math.sin(th)
                for M, byrow in ((A, False), (A, True), (V, False)): ## rotate cols, rows, basis
                    for k in range(p):
                        a, b = (M[i][k], M[j][k]) if byrow else (M[k][i], M[k][j])
                        a, b = c * a - s * b, s * a + c * b
                        if byrow: M[i][k], M[j][k] = a, b
                        else: M[k][i], M[k][j] = a, b
    vals = [A[i][i] for i in range(p)]
    vecs = [[V[k][i] for k in range(p)] for i in range(p)] ## vecs[i] = eigenvector i
    order = sorted(range(p), key=lambda i: -vals[i])
    vals, vecs = [vals[i] for i in order], [vecs[i] for i in order]
    for v in vecs: ## sklearn sign rule: largest |loading| positive
        if v[max(range(p), key=lambda k: abs(v[k]))] < 0: v[:] = [-x for x in v]
    return vals, vecs

def pick_k(ratio, target):
    cum, elbow = 0, len(ratio)
    for i in range(1, len(ratio) - 1): ## first drop that flattens
        if ratio[i] - ratio[i + 1] < 0.5 * (ratio[i - 1] - ratio[i]): elbow = i + 1; break
    for i, r in enumerate(ratio, 1):
        cum += r
        if cum >= target - 1e-12: return elbow, i
    return elbow, len(ratio)

def pca_from_cov(C, names, keep, target, scaled=None, qs=(), scores=True):
    p = len(C)
    hr(f"STEP 3: covariance matrix ({p}x{p})")
    table([""] + names, [[names[i]] + C[i] for i in range(p)])
    print("diagonal = variance, off diagonal = covariance (tilt)")

    vals, vecs = eig_sym(C)
    pcs = [f"PC{i + 1}" for i in range(p)]
    hr("STEP 4: eigenvectors and eigenvalues (C v = lambda v)")
    table(["loading"] + pcs, [[names[i]] + [vecs[j][i] for j in range(p)] for i in range(p)])
    table(["lambda"] + pcs, [["var"] + vals])
    for j in range(p): print(f"{pcs[j]}: C v = {lst(matvec(C, vecs[j]))}  lambda v = {lst([vals[j] * x for x in vecs[j]])}  |v| = {num(norm(vecs[j]))}")
    if p > 1: print(f"PC1 . PC2 = {num(dot(vecs[0], vecs[1]))} (orthogonal)")

    hr("STEP 5: explained variance (scree)")
    tot, cum, rows = sum(vals), 0, []
    ratio = [v / tot for v in vals]
    for j in range(p):
        cum += ratio[j]
        rows.append([pcs[j], vals[j], f"{ratio[j] * 100:.1f}%", f"{cum * 100:.1f}%"])
    table(["", "lambda", "ratio", "cumulative"], rows)
    print(f"total variance = {num(tot)} = trace(C)")
    elbow, hit = pick_k(ratio, target)
    print(f"elbow rule -> keep {elbow}; {target * 100:.0f}% target -> keep {hit}")
    k = keep or hit
    print(f"keeping {k} component(s), retains {sum(ratio[:k]) * 100:.1f}% of variance")

    hr("STEP 6: loadings of kept PCs (weights per feature)")
    for j in range(k): print(f"{pcs[j]} = " + " + ".join(f"({num(vecs[j][i])})*{names[i]}" for i in range(p)))

    if scaled and scores:
        hr(f"STEP 7: project data onto PC1..PC{k} (score = x_scaled . v)")
        table(["#"] + pcs[:k], [[pid(i)] + [dot([c[i] for c in scaled], vecs[j]) for j in range(k)] for i in range(len(scaled[0]))])
    for q in qs:
        hr("QUERY: projected point")
        print(f"scaled query = {lst(q)}")
        for j in range(k): print(f"{pcs[j]} = " + " + ".join(f"{num(q[i])}*{num(vecs[j][i])}" for i in range(p)) + f" = {num(dot(q, vecs[j]))}")
    return vals, vecs

def run_vectors(vs):
    if len({len(v) for v in vs}) > 1: sys.exit("vectors need the same length")
    hr("VECTORS: norm, dot, angle, projection")
    for i, v in enumerate(vs): print(f"v{i} = {lst(v)}  norm = sqrt({' + '.join(f'{num(x)}^2' for x in v)}) = {num(norm(v))}")
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)): ## every pair
            a, b = vs[i], vs[j]
            d, u, na, nb = dot(a, b), unit(b), norm(a), norm(b)
            print(f"\nv{i} . v{j} = {' + '.join(f'{num(x)}*{num(y)}' for x, y in zip(a, b))} = {num(d)}" + ("  (orthogonal)" if abs(d) < 1e-9 else ""))
            cos = d / (na * nb) if na and nb else 0.0
            print(f"cos(theta) = {num(d)} / ({num(na)} * {num(nb)}) = {num(cos)}  theta = {num(math.degrees(math.acos(max(-1, min(1, cos)))))} deg")
            print(f"unit(v{j}) = {lst(u)}  projection of v{i} onto v{j} = v{i} . unit(v{j}) = {num(dot(a, u))}")

def run_data(rows, a):
    hr("STEP 1: data set")
    table(["#"] + FEATURES, [[pid(i)] + list(r) for i, r in enumerate(rows)])
    cols = [list(c) for c in zip(*rows)]
    scaled, stats = scale(cols, a.scale, a.ddof)
    hr(f"STEP 2: scale = {a.scale}" + (f" (ddof={a.ddof})" if a.scale == "zscore" else ""))
    if a.scale != "none":
        lbl = {"center": ("mean", "1"), "zscore": ("mean", "std"), "minmax": ("min", "range"), "robust": ("q1", "iqr")}[a.scale]
        table(["feature"] + list(lbl), [[f, s[1], s[2]] for f, s in zip(FEATURES, stats)])
        print(f"x' = (x - {lbl[0]}) / {lbl[1]}")
        table(["#"] + FEATURES, [[pid(i)] + [c[i] for c in scaled] for i in range(len(rows))])
    else: print("raw values, centered inside the covariance formula only")

    mu = [mean(c) for c in scaled] ## scores need centered data, as sklearn
    centered = [[x - m for x in c] for c, m in zip(scaled, mu)]
    if a.scale in ("minmax", "robust", "none"): print(f"scores below subtract scaled means {lst(mu)} first")
    qs = []
    for q in (parse_matrix(a.query) if a.query else []):
        if len(q) != len(FEATURES): sys.exit(f"query needs {len(FEATURES)} values")
        qs.append([(x - s[1]) / s[2] - m if s[2] else 0.0 for x, s, m in zip(q, stats, mu)])
    pca_from_cov(cov_matrix(scaled, a.ddof), FEATURES, a.keep, a.target, centered, qs, not a.no_scores)

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def interactive(ap):
    mode, argv = ask("mode: data / cov / vec", "data"), []
    if mode == "vec":
        while True:
            v = ask(f"vector {len(argv) // 2} (blank ends)")
            if not v: break
            argv += ["--vec", v]
        return ap.parse_args(argv)
    if mode == "cov":
        argv += ["--cov", ask("covariance or correlation matrix, rows ; separated")]
        argv += ["--names", ask("feature names, comma separated")]
    else:
        argv += ["-d", ask("data: csv path, - for stdin, or rows a,b;c,d")]
        for flag, q in (("--id", "id column name or index (blank = none)"), ("--features", "feature columns (blank = every numeric column)"), ("-q", "query points to project x,y;x,y (blank skips)")):
            v = ask(q)
            if v: argv += [flag, v]
        argv += ["--scale", ask("scale: zscore / center / minmax / robust / none", "zscore")]
        argv += ["--ddof", ask("ddof: 1 sample or 0 population", "1")]
        if ask("show the score table? Y/n", "y").lower() == "n": argv.append("--no-scores")
    k = ask("components to keep (blank = use target)")
    if k: argv += ["--keep", k]
    argv += ["--target", ask("variance target", "0.9")]
    return ap.parse_args(argv)

def main():
    ap = argparse.ArgumentParser(description="PCA on any data set, step by step")
    ap.add_argument("-d", "--data", help="csv file, - for stdin, or inline rows a,b;c,d (header optional)")
    ap.add_argument("--id", help="id column name or index, labels the score rows")
    ap.add_argument("--features", help="feature columns, default every numeric column")
    ap.add_argument("--scale", choices=["zscore", "center", "minmax", "robust", "none"], default="zscore")
    ap.add_argument("--ddof", type=int, choices=[0, 1], default=1, help="1 = sample, 0 = population")
    ap.add_argument("--keep", type=int, help="components to keep, default from target")
    ap.add_argument("--target", type=float, default=0.9, help="cumulative variance target")
    ap.add_argument("-q", "--query", help="points to project x,y;x,y")
    ap.add_argument("--no-scores", action="store_true", help="skip the projection table")
    ap.add_argument("--cov", help="skip data, start from a covariance or correlation matrix a,b;c,d")
    ap.add_argument("--names", help="feature names for --cov")
    ap.add_argument("--vec", action="append", help="vector helper, repeat for pairs")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()

    if a.vec: return run_vectors([parse_csv(v) for v in a.vec])
    if a.cov:
        C = parse_matrix(a.cov)
        p = len(C)
        if any(len(r) != p for r in C) or any(abs(C[i][j] - C[j][i]) > 1e-9 for i in range(p) for j in range(p)): sys.exit("matrix must be square and symmetric")
        names = a.names.split(",") if a.names else [f"x{i}" for i in range(p)]
        if len(names) != p: sys.exit("names count must match matrix size")
        if a.keep and not 0 < a.keep <= p: sys.exit(f"keep must be 1..{p}")
        return pca_from_cov(C, names, a.keep, a.target)
    if not a.data: sys.exit("need -d data, --cov, or --vec")
    load(a.data, a.features, a.id)
    if a.keep and not 0 < a.keep <= len(FEATURES): sys.exit(f"keep must be 1..{len(FEATURES)}")
    if len(DATA) < 2: sys.exit("need at least 2 rows")
    run_data(DATA, a)

if __name__ == "__main__": main()
