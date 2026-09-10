## University of Pittsburgh - CMPINF 2120 Applied Predictive Modeling
## k-means and DBSCAN general use calculator, stdlib only
## general fork of kmeans_dbscan_calc.py: any csv, any dimension, named points
## no flags = prompts, -h = flags

import argparse, math, os, random, re, sys

FEATURES, DATA, IDS = [], [], None ## filled by load()
BASE, METRIC = 0, "euclidean" ## numbering start, dbscan distance

def hr(t): print(f"\n{t}")
def num(v, d=4): return f"{v:.{d}f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)
def row(cells, w=10): return "".join(num(c).rjust(w) for c in cells)
def table(head, rows, w=10): print(row(head, w)); [print(row(r, w)) for r in rows]
def lst(v): return "(" + ", ".join(num(x) for x in v) + ")"
def dbg(tag, v): print(f"[dbg] {tag}: {v}")
def pid(i): return IDS[i] if IDS else i + BASE ## display id

def pidx(tok): ## display id back to index
    try: return IDS.index(tok) if IDS else int(tok) - BASE
    except ValueError: sys.exit(f"no point {tok}")

def parse_csv(s, cast=float):
    try: return [cast(x) for x in s.split(",")]
    except ValueError: sys.exit(f"bad list: {s}")

def isnum(s):
    try: float(s); return True
    except ValueError: return False

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
    if not f: sys.exit("no numeric column found")
    if idj in f: sys.exit("id column cannot be a feature")
    skip = [names[j] for j in range(len(names)) if j not in f and j != idj]
    if skip: print(f"unused column(s): {', '.join(skip)}")
    try: DATA = [tuple(float(r[j]) for j in f) for r in rows]
    except ValueError: sys.exit("non numeric cell in a feature column")
    FEATURES, IDS = [names[j] for j in f], [r[idj] for r in rows] if idj >= 0 else None
    return DATA

def dist(a, b):
    match METRIC:
        case "manhattan": return sum(abs(x - y) for x, y in zip(a, b))
        case "chebyshev": return max(abs(x - y) for x, y in zip(a, b))
        case _: return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def mean_pt(pts): return [sum(c) / len(pts) for c in zip(*pts)]
def nearest(p, cents): return min(range(len(cents)), key=lambda j: (dist(p, cents[j]), j)) ## tie = lower id

def zscore(pts):
    cols = list(zip(*pts))
    mu = [sum(c) / len(c) for c in cols]
    sd = [math.sqrt(sum((x - m) ** 2 for x in c) / len(c)) for c, m in zip(cols, mu)]
    return [[(x - m) / s if s else 0.0 for x, m, s in zip(p, mu, sd)] for p in pts], mu, sd

def init_centroids(pts, k, how, spec, seed, verbose=True):
    rng, n = random.Random(seed), len(pts)
    verbose and hr(f"INIT: {how}" + (f" seed={seed}" if how in ("random", "kmeans++", "farthest") else ""))
    match how:
        case "index":
            idx = [pidx(t.strip()) for t in spec.split(",")]
            if len(idx) != k or any(i < 0 or i >= n for i in idx): sys.exit(f"need {k} valid point ids")
        case "coords":
            cents = [parse_csv(c) for c in spec.split(";")]
            if len(cents) != k or any(len(c) != len(FEATURES) for c in cents): sys.exit(f"need {k} centroids as x,y;x,y")
            for j, c in enumerate(cents): verbose and print(f"C{j} = {lst(c)} (arbitrary point in space)")
            return cents
        case "random": idx = rng.sample(range(n), k)
        case _: ## kmeans++ or farthest
            idx = [rng.randrange(n)]
            verbose and print(f"C0 = point {pid(idx[0])} picked at random")
            while len(idx) < k:
                d2 = [min(dist(p, pts[i]) for i in idx) ** 2 for p in pts]
                tot = sum(d2)
                verbose and table(["#", "D^2", "prob"], [[pid(i), d2[i], d2[i] / tot if tot else 0] for i in range(n) if i not in idx])
                if how == "farthest": pick = max(range(n), key=lambda i: (d2[i], -i))
                else:
                    r, acc, pick = rng.random() * tot, 0, n - 1
                    for i in range(n):
                        acc += d2[i]
                        if acc >= r: pick = i; break
                verbose and print(f"C{len(idx)} = point {pid(pick)} ({'farthest' if how == 'farthest' else 'drawn by D^2 weight'})")
                idx.append(pick)
    for j, i in enumerate(idx): verbose and print(f"C{j} = point {pid(i)} {lst(pts[i])}")
    return [list(pts[i]) for i in idx]

def assign(pts, cents, verbose):
    lab = [nearest(p, cents) for p in pts]
    if verbose:
        table(["#"] + [f"d(C{j})" for j in range(len(cents))] + ["cluster"], [[pid(i)] + [dist(p, c) for c in cents] + [f"C{lab[i]}"] for i, p in enumerate(pts)])
    return lab

def update(pts, cents, lab, verbose):
    new = []
    for j, c in enumerate(cents):
        members = [p for p, l in zip(pts, lab) if l == j]
        if not members: new.append(c); verbose and print(f"C{j}: empty, centroid kept"); continue
        m = mean_pt(members)
        if verbose:
            ids = ", ".join(str(pid(i)) for i, l in enumerate(lab) if l == j)
            print(f"C{j} = mean of {{{ids}}} = " + lst([sum(col) for col in zip(*members)]) + f" / {len(members)} = {lst(m)}")
        new.append(m)
    return new

def kmeans(pts, cents, max_iter, verbose=True):
    lab = None
    for it in range(1, max_iter + 1):
        verbose and hr(f"ITERATION {it}: assign each point to nearest centroid")
        new_lab = assign(pts, cents, verbose)
        verbose and hr(f"ITERATION {it}: move centroids to cluster means")
        new_cents = update(pts, cents, new_lab, verbose)
        moved = max(dist(a, b) for a, b in zip(cents, new_cents))
        changed = sum(a != b for a, b in zip(lab, new_lab)) if lab else len(pts)
        verbose and print(f"reassigned = {changed}, max centroid move = {num(moved)}")
        lab, cents = new_lab, new_cents
        if changed == 0 and moved < 1e-12:
            verbose and print(f"converged at iteration {it}")
            return lab, cents, it
    verbose and print(f"timeout after {max_iter} iterations")
    return lab, cents, max_iter

def wcss(pts, cents, lab, verbose=True):
    tot, rows = 0, []
    for j, c in enumerate(cents):
        sq = [dist(p, c) ** 2 for p, l in zip(pts, lab) if l == j]
        tot += sum(sq)
        rows.append([f"C{j}", len(sq), sum(sq)])
    if verbose:
        hr("WCSS = sum over clusters, sum over points of d(point, centroid)^2")
        table(["cluster", "n", "sum d^2"], rows)
        print(f"WCSS = {' + '.join(num(r[2]) for r in rows)} = {num(tot)}")
    return tot

def silhouette(pts, lab, ids=None): ## ids = display ids override
    hr("SILHOUETTE: s = (b - a) / max(a, b)")
    k, rows, scores = max(lab) + 1, [], []
    if k < 2: print("needs k >= 2"); return None
    for i, p in enumerate(pts):
        own = [dist(p, q) for j, q in enumerate(pts) if j != i and lab[j] == lab[i]]
        others = [sum(dist(p, q) for j, q in enumerate(pts) if lab[j] == c) / lab.count(c) for c in range(k) if c != lab[i] and lab.count(c)]
        a, b = (sum(own) / len(own) if own else 0.0), min(others)
        s = (b - a) / max(a, b) if own and max(a, b) else 0.0 ## singleton = 0, sklearn rule
        scores.append(s)
        rows.append([ids[i] if ids else pid(i), f"C{lab[i]}", a, b, s])
    table(["#", "cluster", "a (own)", "b (next)", "s"], rows)
    print(f"mean silhouette = {num(sum(scores))} / {len(scores)} = {num(sum(scores) / len(scores))}  (+1 tight, 0 boundary, -1 wrong cluster)")
    return sum(scores) / len(scores)

def runs(pts, k, how, seed, n, max_iter): ## quiet restarts, (seed, wcss)
    how, out = how if how in ("random", "kmeans++", "farthest") else "random", []
    for s in range(seed, seed + n):
        lab, cents, _ = kmeans(pts, init_centroids(pts, k, how, None, s, False), max_iter, False)
        out.append((s, wcss(pts, cents, lab, False)))
    return out

def best_seed(pts, k, how, seed, n, max_iter):
    hr(f"RESTARTS: {n} runs, seeds {seed}..{seed + n - 1}, keep the lowest WCSS")
    r = runs(pts, k, how, seed, n, max_iter)
    table(["seed", "WCSS"], [list(t) for t in r], 12)
    best = min(r, key=lambda t: (t[1], t[0]))
    print(f"best seed = {best[0]} (WCSS {num(best[1])})")
    return best[0]

def elbow(pts, kmax, how, seed, max_iter):
    hr(f"ELBOW: WCSS for k = 1..{kmax}, best of 10 restarts each")
    rows, prev = [], None
    for k in range(1, kmax + 1):
        best = min(w for _, w in runs(pts, k, how, seed, 10, max_iter))
        rows.append([k, best, "" if prev is None else prev - best])
        prev = best
    table(["k", "WCSS", "drop"], rows, 12)
    drops = [r[2] for r in rows[1:]]
    for i in range(1, len(drops)): ## first drop under half the previous
        if drops[i] < 0.5 * drops[i - 1]: print(f"elbow rule -> k = {i + 1} (drop after it shrinks to {num(drops[i])})"); return
    print("no clear elbow in this range")

## DBSCAN
def ids(s): return "{" + ", ".join(str(pid(i)) for i in sorted(s)) + "}"

def neighbors(pts, eps): return [[j for j in range(len(pts)) if j != i and dist(pts[i], pts[j]) <= eps] for i in range(len(pts))]

def classify(nb, min_pts, self_counts):
    core = {i for i in range(len(nb)) if len(nb[i]) + self_counts >= min_pts}
    border = {i for i in range(len(nb)) if i not in core and any(j in core for j in nb[i])}
    return core, border, set(range(len(nb))) - core - border

def dbscan(pts, eps, min_pts, self_counts, verbose=True):
    nb = neighbors(pts, eps)
    core, border, noise = classify(nb, min_pts, self_counts)
    if verbose:
        hr(f"STEP 3: neighbors within eps = {num(float(eps))} ({METRIC} distance <= eps)")
        print(f"minPts = {min_pts}, a point {'counts' if self_counts else 'does not count'} itself")
        print(f"{'#':>6}  {'count':>5}  {'type':<6}  neighbors")
        for i in range(len(pts)): print(f"{str(pid(i)):>6}  {len(nb[i]) + self_counts:>5}  {'core' if i in core else 'border' if i in border else 'noise':<6}  {ids(nb[i])}")
        print("core = count >= minPts, border = not core but touches a core, noise = neither")

    lab, k, log = [-1] * len(pts), 0, []
    for s in sorted(core): ## flood fill from each unclaimed core
        if lab[s] >= 0: continue
        lab[s], stack, grew = k, [s], [s]
        while stack:
            c = stack.pop(0)
            for j in nb[c]:
                if lab[j] >= 0: continue
                lab[j] = k
                grew.append(j)
                if j in core: stack.append(j) ## only cores keep expanding
        log.append((k, s, grew))
        k += 1
    if verbose:
        hr("STEP 4: grow clusters from cores (border joins first cluster to reach it)")
        for c, s, grew in log: print(f"C{c}: seed {pid(s)} -> reach order [{', '.join(str(pid(i)) for i in grew)}]")
        if border: print(f"relabeled: {ids(border)} were noise on their own visit, a core reached them, now border")
        shared = [i for i in border if len({lab[j] for j in nb[i] if j in core}) > 1]
        if shared: print(f"NOTE: border point(s) {ids(shared)} touch more than one cluster, order decided")
    return lab, core, border, noise

def db_report(pts, lab, core, border, noise, sil):
    k = max(lab) + 1
    hr(f"RESULT: {k} cluster(s), {len(noise)} noise")
    for c in range(k):
        m = {i for i in range(len(pts)) if lab[i] == c}
        print(f"C{c} ({len(m)}): cores {ids(m & core)}  borders {ids(m & border)}")
    print(f"noise: {ids(noise)}")
    if sil and k >= 2:
        keep = [i for i in range(len(pts)) if lab[i] >= 0]
        print("(silhouette over clustered points only, noise dropped)")
        silhouette([pts[i] for i in keep], [lab[i] for i in keep], [pid(i) for i in keep])

def kdist(pts, min_pts, self_counts):
    kth = min_pts - self_counts ## distance to the kth other point
    hr(f"K-DISTANCE: distance to nearest other point number {kth}, sorted (elbow = eps)")
    d = sorted(sorted(dist(p, q) for q in pts if q is not p)[kth - 1] for p in pts) if 0 < kth < len(pts) else []
    if not d: print("minPts too big for this data"); return
    table(["rank", "kdist"], [[r, v] for r, v in enumerate(d, 1)], 10)
    jumps = [d[i] - d[i - 1] for i in range(1, len(d))]
    if jumps: print(f"largest jump after rank {jumps.index(max(jumps)) + 1}: eps around {num(d[jumps.index(max(jumps))])} to {num(d[jumps.index(max(jumps)) + 1])}")

def eps_sweep(pts, eps_list, min_pts, self_counts):
    hr(f"SWEEP: eps in {[num(float(e)) for e in eps_list]} with minPts = {min_pts}")
    rows = []
    for e in eps_list:
        lab, core, border, noise = dbscan(pts, e, min_pts, self_counts, verbose=False)
        rows.append([num(float(e)), max(lab) + 1, len(core), len(border), len(noise)])
    table(["eps", "clusters", "core", "border", "noise"], rows)

def ask(label, default=""): ## Enter keeps the default
    v = input(f"{label}{f' [{default}]' if default else ''}: ").strip()
    return v or default

def interactive(ap):
    argv = ["-d", ask("data: csv path, - for stdin, or rows x,y;x,y")]
    for flag, q in (("--id", "id column name or index (blank = numbered)"), ("--features", "feature columns (blank = every numeric column)")):
        v = ask(q)
        if v: argv += [flag, v]
    algo = ask("algorithm: kmeans / dbscan", "kmeans")
    argv += ["--algo", algo, "--base", ask("point numbering starts at 0 or 1", str(BASE))]
    if algo == "dbscan":
        argv += ["--eps", ask("eps (radius)", "1.5"), "--min-pts", ask("minPts", "3"), "--metric", ask("metric euclidean/manhattan/chebyshev", "euclidean")]
        if ask("does a point count itself toward minPts? Y/n", "y").lower() == "n": argv.append("--no-self")
    else:
        argv += ["-k", ask("k", "3")]
        how = ask("init: index / coords / random / kmeans++ / farthest", "random")
        argv += ["--init", how]
        if how == "index": argv += ["--spec", ask("point ids for initial centroids, comma separated")]
        elif how == "coords": argv += ["--spec", ask("centroid coords x,y;x,y")]
        else: argv += ["--seed", ask("seed", "0"), "--restarts", ask("restarts (lowest WCSS wins)", "1")]
        argv += ["--max-iter", ask("max iterations (timeout)", "100")]
    if ask("standardize features? y/N", "n").lower() == "y": argv.append("--std")
    if ask("silhouette? Y/n", "y").lower() == "n": argv.append("--no-sil")
    if algo == "dbscan":
        if ask("k-distance table for picking eps? y/N", "n").lower() == "y": argv.append("--kdist")
        s = ask("sweep eps values, e.g. 1,1.5,2 (blank skips)")
        if s: argv += ["--sweep", s]
    else:
        e = ask("elbow sweep up to k = (blank skips)")
        if e: argv += ["--elbow", e]
    return ap.parse_args(argv)

def main():
    global BASE, METRIC
    ap = argparse.ArgumentParser(description="k-means or DBSCAN on any data set, step by step")
    ap.add_argument("-d", "--data", help="csv file, - for stdin, or inline rows x,y;x,y (header optional)")
    ap.add_argument("--id", help="id column name or index, points print by name")
    ap.add_argument("--features", help="feature columns, default every numeric column")
    ap.add_argument("--algo", choices=["kmeans", "dbscan"], default="kmeans")
    ap.add_argument("--std", action="store_true", help="z-score features first")
    ap.add_argument("--no-sil", action="store_true", help="skip silhouette")
    ap.add_argument("--base", type=int, choices=[0, 1], default=BASE, help="point numbering start when no --id")
    ap.add_argument("--metric", choices=["euclidean", "manhattan", "chebyshev"], default="euclidean", help="dbscan distance, kmeans is always euclidean")
    ap.add_argument("--brief", action="store_true", help="results only, skip the step by step")
    ap.add_argument("-k", type=int, default=3, help="kmeans: clusters")
    ap.add_argument("--init", choices=["index", "coords", "random", "kmeans++", "farthest"], default="random")
    ap.add_argument("--spec", help="index: point ids a,b,c  coords: 1,2;5,6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=1, help="kmeans: seeded runs, lowest WCSS wins")
    ap.add_argument("--elbow", type=int, help="kmeans: also sweep WCSS for k = 1..N")
    ap.add_argument("--eps", type=float, default=1.5, help="dbscan: neighborhood radius")
    ap.add_argument("--min-pts", type=int, default=3, help="dbscan: minPts")
    ap.add_argument("--no-self", action="store_true", help="dbscan: point does not count itself")
    ap.add_argument("--kdist", action="store_true", help="dbscan: k-distance table for choosing eps")
    ap.add_argument("--sweep", help="dbscan: eps values to compare, e.g. 1,1.5,2")
    a = interactive(ap) if len(sys.argv) == 1 else ap.parse_args()
    if not a.data: sys.exit("need -d data")
    load(a.data, a.features, a.id)
    BASE, METRIC = a.base, a.metric if a.algo == "dbscan" else "euclidean"

    pts, v = [list(p) for p in DATA], not a.brief
    if v:
        hr("STEP 1: data set")
        table(["#"] + FEATURES, [[pid(i)] + list(p) for i, p in enumerate(pts)])
    if a.std:
        pts, mu, sd = zscore(pts)
        if v:
            hr("STEP 2: standardize (z = (x - mean) / std, ddof=0)")
            table(["feature", "mean", "std"], [[f, m, s] for f, m, s in zip(FEATURES, mu, sd)])
            table(["#"] + FEATURES, [[pid(i)] + p for i, p in enumerate(pts)])

    if a.algo == "dbscan":
        self_counts = 0 if a.no_self else 1
        if a.eps <= 0 or a.min_pts < 1: sys.exit("eps must be > 0 and minPts >= 1")
        lab, core, border, noise = dbscan(pts, a.eps, a.min_pts, self_counts, v)
        db_report(pts, lab, core, border, noise, not a.no_sil)
        if a.kdist: kdist(pts, a.min_pts, self_counts)
        if a.sweep: eps_sweep(pts, parse_csv(a.sweep), a.min_pts, self_counts)
        return
    if not 0 < a.k <= len(pts): sys.exit(f"k must be 1..{len(pts)}")
    if a.init in ("index", "coords") and not a.spec: a.spec = ",".join(str(pid(i)) for i in range(a.k)) if a.init == "index" else sys.exit("coords init needs --spec")
    if a.restarts > 1 and a.init not in ("index", "coords"): a.seed = best_seed(pts, a.k, a.init, a.seed, a.restarts, a.max_iter)
    cents = init_centroids(pts, a.k, a.init, a.spec, a.seed, v)
    lab, cents, it = kmeans(pts, cents, a.max_iter, v)
    hr(f"RESULT after {it} iteration(s)")
    for j, c in enumerate(cents): print(f"C{j} {lst(c)}: points {{{', '.join(str(pid(i)) for i, l in enumerate(lab) if l == j)}}}")
    wcss(pts, cents, lab)
    if not a.no_sil: silhouette(pts, lab)
    if a.elbow: elbow(pts, min(a.elbow, len(pts)), a.init, a.seed, a.max_iter)

if __name__ == "__main__": main()
