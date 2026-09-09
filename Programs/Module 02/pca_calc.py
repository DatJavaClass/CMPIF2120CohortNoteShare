## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## PCA bare metal calculator, stdlib only
## no flags = prompts, -h = flags

import argparse, math, sys

## edit the data set here, one row per observation
FEATURES = ["length_m", "weight_kg"]
DATA = [(2.7, 900), (3.8, 1200), (4.0, 1300), (4.6, 1500), \
        (4.8, 2100), (5.1, 2000), (5.9, 2600), (6.5, 3500)]

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=12): return "".join(num(c).rjust(w) for c in cells)
def table(head, rows, w=12): print(row(head, w)); [print(row(r, w)) for r in rows]
def lst(v): return "[" + ", ".join(num(x) for x in v) + "]"
def dbg(tag, v): print(f"[dbg] {tag}: {v}")

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

def parse_matrix(s): return [parse_csv(r) for r in s.split(";")] ## "a,b;c,d"

## vector helpers, lecture 5
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

def pca_from_cov(C, names, keep, target, scaled=None, q=None):
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

    if scaled:
        hr(f"STEP 7: project data onto PC1..PC{k} (score = x_scaled . v)")
        table(["#"] + pcs[:k], [[i] + [dot([c[i] for c in scaled], vecs[j]) for j in range(k)] for i in range(len(scaled[0]))])
    if q:
        hr("QUERY: projected point")
        print(f"scaled query = {lst(q)}")
        for j in range(k): print(f"{pcs[j]} = " + " + ".join(f"{num(q[i])}*{num(vecs[j][i])}" for i in range(p)) + f" = {num(dot(q, vecs[j]))}")
    return vals, vecs

def run_vectors(vs):
    hr("VECTORS: norm, dot, angle, projection")
    for i, v in enumerate(vs): print(f"v{i} = {lst(v)}  norm = sqrt({' + '.join(f'{num(x)}^2' for x in v)}) = {num(norm(v))}")
    if len(vs) < 2: return
    a, b = vs[0], vs[1]
    d, u = dot(a, b), unit(b)
    print(f"v0 . v1 = {' + '.join(f'{num(x)}*{num(y)}' for x, y in zip(a, b))} = {num(d)}" + ("  (orthogonal)" if abs(d) < 1e-9 else ""))
    cos = d / (norm(a) * norm(b))
    print(f"cos(theta) = {num(d)} / ({num(norm(a))} * {num(norm(b))}) = {num(cos)}  theta = {num(math.degrees(math.acos(max(-1, min(1, cos)))))} deg")
    print(f"unit(v1) = {lst(u)}  projection of v0 onto v1 = v0 . unit(v1) = {num(dot(a, u))}")

def run_data(rows, a):
    hr("STEP 1: data set")
    table(["#"] + FEATURES, [[i] + list(r) for i, r in enumerate(rows)])
    cols = [list(c) for c in zip(*rows)]
    scaled, stats = scale(cols, a.scale, a.ddof)
    hr(f"STEP 2: scale = {a.scale}" + (f" (ddof={a.ddof})" if a.scale == "zscore" else ""))
    if a.scale != "none":
        lbl = {"center": ("mean", "1"), "zscore": ("mean", "std"), "minmax": ("min", "range"), "robust": ("q1", "iqr")}[a.scale]
        table(["feature"] + list(lbl), [[f, s[1], s[2]] for f, s in zip(FEATURES, stats)])
        print(f"x' = (x - {lbl[0]}) / {lbl[1]}")
        table(["#"] + FEATURES, [[i] + [c[i] for c in scaled] for i in range(len(rows))])
    else: print("raw values, centered inside the covariance formula only")

    mu = [mean(c) for c in scaled] ## scores need centered data, as sklearn
    centered = [[x - m for x in c] for c, m in zip(scaled, mu)]
    if a.scale in ("minmax", "robust", "none"): print(f"scores below subtract scaled means {lst(mu)} first")
    q = None
    if a.query:
        q = parse_csv(a.query)
        if len(q) != len(FEATURES): sys.exit(f"query needs {len(FEATURES)} values")
        q = [(x - s[1]) / s[2] - m if s[2] else 0.0 for x, s, m in zip(q, stats, mu)]
    pca_from_cov(cov_matrix(scaled, a.ddof), FEATURES, a.keep, a.target, centered, q)

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def ask_data():
    global FEATURES, DATA
    FEATURES = ask("feature names, comma separated", ",".join(FEATURES)).split(",")
    print(f"rows as {','.join(FEATURES)}; blank line ends")
    rows = []
    while True:
        line = input(f"row {len(rows)}: ").strip()
        if not line: break
        try: rows.append(tuple(float(x) for x in line.split(",")))
        except ValueError: print("  not numeric, retry"); continue
        if len(rows[-1]) != len(FEATURES): rows.pop(); print(f"  need {len(FEATURES)} values, retry")
    if rows: DATA = rows
    else: print("no rows, keeping built in data")

def interactive(ap):
    mode, argv = ask("mode: data / cov / vec", "data"), []
    if mode == "vec":
        while True:
            v = ask(f"vector {len(argv) // 2} (blank ends)")
            if not v: break
            argv += ["--vec", v]
        return ap.parse_args(argv)
    if mode == "cov":
        argv += ["--cov", ask("covariance matrix, rows ; separated", "2,1.3;1.3,2")]
        argv += ["--names", ask("feature names", "x,y")]
    else:
        if ask("edit the data set? y/N", "n").lower() == "y": ask_data()
        argv += ["--scale", ask("scale: zscore / center / minmax / robust / none", "zscore")]
        argv += ["--ddof", ask("ddof: 1 sample (numpy) or 0 population", "1")]
        q = ask("query point to project " + ",".join(FEATURES) + " (blank skips)")
        if q: argv += ["-q", q]
    k = ask("components to keep (blank = use target)")
    if k: argv += ["--keep", k]
    argv += ["--target", ask("variance target", "0.9")]
    return ap.parse_args(argv)

def main():
    ap = argparse.ArgumentParser(description="PCA, step by step")
    ap.add_argument("--scale", choices=["zscore", "center", "minmax", "robust", "none"], default="zscore")
    ap.add_argument("--ddof", type=int, choices=[0, 1], default=1, help="1 = sample (numpy), 0 = population (lecture)")
    ap.add_argument("--keep", type=int, help="components to keep, default from target")
    ap.add_argument("--target", type=float, default=0.9, help="cumulative variance target")
    ap.add_argument("-q", "--query", help="point to project, comma separated")
    ap.add_argument("--cov", help="skip data, start from covariance matrix a,b;c,d")
    ap.add_argument("--names", help="feature names for --cov")
    ap.add_argument("--vec", action="append", help="vector helper, repeat for two")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()

    if a.vec: return run_vectors([parse_csv(v) for v in a.vec])
    if a.cov:
        C = parse_matrix(a.cov)
        p = len(C)
        if any(len(r) != p for r in C) or any(abs(C[i][j] - C[j][i]) > 1e-9 for i in range(p) for j in range(p)): sys.exit("covariance matrix must be square and symmetric")
        names = a.names.split(",") if a.names else [f"x{i}" for i in range(p)]
        if len(names) != p: sys.exit("names count must match matrix size")
        return pca_from_cov(C, names, a.keep, a.target)
    if a.keep and not 0 < a.keep <= len(FEATURES): sys.exit(f"keep must be 1..{len(FEATURES)}")
    if len(DATA) < 3: sys.exit("need at least 3 rows")
    run_data(DATA, a)

if __name__ == "__main__": main()
